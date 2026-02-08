"""CUDA-first trainer for astro tabular neural models."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfig, ScoutConfig, TrainConfig
from .metrics_numba import (
    predict_with_margin,
    predict_with_threshold_binary,
    search_best_margin,
    search_best_threshold_binary,
    summarize_directional_metrics,
)
from .model_dcn import AstroTabularDCN
from .model_deepfm import AstroTabularDeepFM


@dataclass(frozen=True)
class FitResult:
    """Training outputs for experiment table and notebook visualization."""

    model_name: str
    best_epoch: int
    best_val_score: float
    best_margin: float
    cutoff_kind: str
    train_loss_last: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    history: Dict[str, List[float]]
    test_pred: Optional[np.ndarray] = None
    test_proba: Optional[np.ndarray] = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp_dtype(name: str) -> torch.dtype:
    key = str(name).lower().strip()
    if key == "float16":
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this research direction, but no CUDA device is available")

    dev = torch.device(device_name)
    if dev.type != "cuda":
        raise RuntimeError(f"device must be CUDA, got: {device_name}")

    return dev


def _tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(max(num_workers, 0)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(num_workers > 0),
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses: List[float] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def _predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        chunks.append(probs.cpu().numpy())

    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _build_model(model_cfg: ModelConfig, input_dim: int, n_classes: int) -> nn.Module:
    kind = str(model_cfg.model_type).lower().strip()
    if kind == "dcn":
        return AstroTabularDCN(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=model_cfg.hidden_dims,
            cross_layers=model_cfg.cross_layers,
            cross_rank=model_cfg.cross_rank,
            dropout=model_cfg.dropout,
            activation=model_cfg.activation,
        )
    if kind in {"deepfm", "deep_fm", "fm"}:
        return AstroTabularDeepFM(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=model_cfg.hidden_dims,
            embed_dim=model_cfg.embed_dim,
            dropout=model_cfg.dropout,
            activation=model_cfg.activation,
        )
    raise ValueError(f"Unsupported model_type: {model_cfg.model_type}")


def _run_cutoff_search(
    probs: np.ndarray,
    y_true: np.ndarray,
    scout_cfg: ScoutConfig,
    n_classes: int,
) -> Tuple[str, float, float]:
    if n_classes == 2:
        threshold_grid = np.asarray(scout_cfg.threshold_grid, dtype=np.float64)
        res = search_best_threshold_binary(
            probs=probs,
            y_true=y_true,
            thresholds=threshold_grid,
            gap_penalty=float(scout_cfg.margin_gap_penalty),
            prior_penalty=float(scout_cfg.margin_prior_penalty),
        )
        return "threshold", float(res.best_threshold), float(res.best_score)

    margin_grid = np.asarray(scout_cfg.margin_grid, dtype=np.float64)
    res = search_best_margin(
        probs=probs,
        y_true=y_true,
        margins=margin_grid,
        gap_penalty=float(scout_cfg.margin_gap_penalty),
        prior_penalty=float(scout_cfg.margin_prior_penalty),
    )
    return "margin", float(res.best_margin), float(res.best_score)


def _predict_from_cutoff(
    probs: np.ndarray,
    cutoff_kind: str,
    cutoff_value: float,
    n_classes: int,
) -> np.ndarray:
    if n_classes == 2 or str(cutoff_kind) == "threshold":
        return predict_with_threshold_binary(probs=probs, threshold=float(cutoff_value))
    return predict_with_margin(probs=probs, margin=float(cutoff_value))


def fit_dcn_model(
    model_name: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    scout_cfg: ScoutConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weights: np.ndarray,
    capture_predictions: bool = False,
) -> FitResult:
    """Train one model config and evaluate on train/val/test."""
    device = _require_cuda(train_cfg.device)
    _set_seed(train_cfg.seed)

    n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    input_dim = int(X_train.shape[1])

    train_loader = _tensor_loader(
        X=X_train,
        y=y_train,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    eval_loader_train = _tensor_loader(
        X=X_train,
        y=y_train,
        batch_size=max(1024, train_cfg.batch_size),
        shuffle=False,
        num_workers=max(0, train_cfg.num_workers // 2),
    )
    eval_loader_val = _tensor_loader(
        X=X_val,
        y=y_val,
        batch_size=max(1024, train_cfg.batch_size),
        shuffle=False,
        num_workers=max(0, train_cfg.num_workers // 2),
    )
    eval_loader_test = _tensor_loader(
        X=X_test,
        y=y_test,
        batch_size=max(1024, train_cfg.batch_size),
        shuffle=False,
        num_workers=max(0, train_cfg.num_workers // 2),
    )

    model = _build_model(model_cfg=model_cfg, input_dim=input_dim, n_classes=n_classes).to(device)
    if train_cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    weight_tensor = torch.from_numpy(class_weights.astype(np.float32, copy=False)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(train_cfg.label_smoothing))

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    amp_enabled = bool(train_cfg.amp_enabled)
    amp_dtype = _amp_dtype(train_cfg.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_cutoff = 0.0
    best_cutoff_kind = "margin"
    best_val_score = -1e9
    wait = 0

    hist_loss: List[float] = []
    hist_val_score: List[float] = []
    hist_cutoff: List[float] = []

    t0 = time.perf_counter()

    for epoch in range(int(train_cfg.epochs)):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip_norm=float(train_cfg.grad_clip_norm),
        )

        probs_val = _predict_proba(model=model, loader=eval_loader_val, device=device)
        cutoff_kind, cutoff, score = _run_cutoff_search(
            probs=probs_val,
            y_true=y_val,
            scout_cfg=scout_cfg,
            n_classes=n_classes,
        )

        hist_loss.append(train_loss)
        hist_val_score.append(float(score))
        hist_cutoff.append(float(cutoff))

        if score > best_val_score:
            best_val_score = float(score)
            best_cutoff = float(cutoff)
            best_cutoff_kind = str(cutoff_kind)
            best_epoch = int(epoch + 1)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(train_cfg.early_stopping_patience):
                break

    model.load_state_dict(best_state)

    probs_train = _predict_proba(model=model, loader=eval_loader_train, device=device)
    probs_val = _predict_proba(model=model, loader=eval_loader_val, device=device)
    probs_test = _predict_proba(model=model, loader=eval_loader_test, device=device)

    pred_train = _predict_from_cutoff(
        probs=probs_train,
        cutoff_kind=best_cutoff_kind,
        cutoff_value=best_cutoff,
        n_classes=n_classes,
    )
    pred_val = _predict_from_cutoff(
        probs=probs_val,
        cutoff_kind=best_cutoff_kind,
        cutoff_value=best_cutoff,
        n_classes=n_classes,
    )
    pred_test = _predict_from_cutoff(
        probs=probs_test,
        cutoff_kind=best_cutoff_kind,
        cutoff_value=best_cutoff,
        n_classes=n_classes,
    )

    metrics_train = summarize_directional_metrics(y_true=y_train, y_pred=pred_train, n_classes=n_classes)
    metrics_val = summarize_directional_metrics(y_true=y_val, y_pred=pred_val, n_classes=n_classes)
    metrics_test = summarize_directional_metrics(y_true=y_test, y_pred=pred_test, n_classes=n_classes)

    elapsed = time.perf_counter() - t0

    history = {
        "loss": hist_loss,
        "val_score": hist_val_score,
        "margin": hist_cutoff,
        "elapsed_sec": [float(elapsed)],
    }

    return FitResult(
        model_name=model_name,
        best_epoch=best_epoch,
        best_val_score=best_val_score,
        best_margin=best_cutoff,
        cutoff_kind=best_cutoff_kind,
        train_loss_last=float(hist_loss[-1]) if hist_loss else float("nan"),
        train_metrics=metrics_train,
        val_metrics=metrics_val,
        test_metrics=metrics_test,
        history=history,
        test_pred=pred_test.copy() if capture_predictions else None,
        test_proba=probs_test.copy() if capture_predictions else None,
    )
