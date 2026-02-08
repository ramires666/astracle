"""Broad grid-search trial runner for astro tabular neural experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ModelConfig, ScoutConfig
from .data_utils import LoadedDataset, build_time_split, compute_class_weights, split_summary
from .trainer import fit_dcn_model


@dataclass(frozen=True)
class GridSearchSpace:
    """Broad search ranges for first trial sweep."""

    model_types: Tuple[str, ...] = ("dcn", "deepfm")

    hidden_dims: Tuple[Tuple[int, ...], ...] = (
        (256, 128),
        (384, 192, 96),
        (512, 256, 128),
        (640, 320, 160),
    )
    dropout: Tuple[float, ...] = (0.10, 0.15, 0.25, 0.35)

    # DCN-specific
    cross_layers: Tuple[int, ...] = (2, 3, 5)
    cross_rank: Tuple[int, ...] = (32, 64, 96)

    # DeepFM-specific
    embed_dim: Tuple[int, ...] = (16, 32, 64)

    # Shared train hypers
    learning_rate: Tuple[float, ...] = (5e-4, 1e-3, 2e-3)
    weight_decay: Tuple[float, ...] = (1e-5, 1e-4, 5e-4)
    class_weight_power: Tuple[float, ...] = (1.0, 1.2, 1.4)
    label_smoothing: Tuple[float, ...] = (0.0, 0.02, 0.05)
    batch_size: Tuple[int, ...] = (384, 512, 768)


@dataclass(frozen=True)
class GridRunSpec:
    """One concrete candidate sampled from search space."""

    model_type: str
    hidden_dims: Tuple[int, ...]
    dropout: float
    cross_layers: int
    cross_rank: int
    embed_dim: int
    learning_rate: float
    weight_decay: float
    class_weight_power: float
    label_smoothing: float
    batch_size: int


@dataclass(frozen=True)
class ScaledSplit:
    """Chronological split with scaled arrays shared across runs."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_candidate_pool(space: GridSearchSpace) -> List[GridRunSpec]:
    """Create full candidate pool from architecture-aware products."""
    specs: List[GridRunSpec] = []

    for model_type in space.model_types:
        mt = str(model_type).lower().strip()

        if mt == "dcn":
            for (
                hidden_dims,
                dropout,
                cross_layers,
                cross_rank,
                learning_rate,
                weight_decay,
                class_weight_power,
                label_smoothing,
                batch_size,
            ) in product(
                space.hidden_dims,
                space.dropout,
                space.cross_layers,
                space.cross_rank,
                space.learning_rate,
                space.weight_decay,
                space.class_weight_power,
                space.label_smoothing,
                space.batch_size,
            ):
                specs.append(
                    GridRunSpec(
                        model_type="dcn",
                        hidden_dims=tuple(int(v) for v in hidden_dims),
                        dropout=float(dropout),
                        cross_layers=int(cross_layers),
                        cross_rank=int(cross_rank),
                        embed_dim=0,
                        learning_rate=float(learning_rate),
                        weight_decay=float(weight_decay),
                        class_weight_power=float(class_weight_power),
                        label_smoothing=float(label_smoothing),
                        batch_size=int(batch_size),
                    )
                )
            continue

        if mt == "deepfm":
            for (
                hidden_dims,
                dropout,
                embed_dim,
                learning_rate,
                weight_decay,
                class_weight_power,
                label_smoothing,
                batch_size,
            ) in product(
                space.hidden_dims,
                space.dropout,
                space.embed_dim,
                space.learning_rate,
                space.weight_decay,
                space.class_weight_power,
                space.label_smoothing,
                space.batch_size,
            ):
                specs.append(
                    GridRunSpec(
                        model_type="deepfm",
                        hidden_dims=tuple(int(v) for v in hidden_dims),
                        dropout=float(dropout),
                        cross_layers=0,
                        cross_rank=0,
                        embed_dim=int(embed_dim),
                        learning_rate=float(learning_rate),
                        weight_decay=float(weight_decay),
                        class_weight_power=float(class_weight_power),
                        label_smoothing=float(label_smoothing),
                        batch_size=int(batch_size),
                    )
                )
            continue

        raise ValueError(f"Unsupported model type in search space: {model_type}")

    return specs


def sample_candidates(
    pool: Sequence[GridRunSpec],
    n_trials: int,
    seed: int,
) -> List[GridRunSpec]:
    """Sample unique candidates for trial search."""
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if len(pool) == 0:
        raise ValueError("candidate pool is empty")

    if n_trials >= len(pool):
        return list(pool)

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(pool), size=int(n_trials), replace=False)
    return [pool[int(i)] for i in idx.tolist()]


def _make_scaled_split(dataset: LoadedDataset, scout_cfg: ScoutConfig) -> Tuple[ScaledSplit, Dict[str, object]]:
    """Prepare scaled arrays once for all candidates."""
    split = build_time_split(n_rows=len(dataset.y), cfg=scout_cfg.split)
    meta = split_summary(dataset=dataset, split=split)

    X_train_raw = dataset.X_raw[split.train_idx]
    X_val_raw = dataset.X_raw[split.val_idx]
    X_test_raw = dataset.X_raw[split.test_idx]

    y_train = dataset.y[split.train_idx]
    y_val = dataset.y[split.val_idx]
    y_test = dataset.y[split.test_idx]

    # Robust scaling once (shared for all runs).
    q_low, q_high = 5.0, 95.0
    med = np.median(X_train_raw, axis=0)
    q1 = np.percentile(X_train_raw, q_low, axis=0)
    q2 = np.percentile(X_train_raw, q_high, axis=0)
    scale = q2 - q1
    scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)

    X_train = ((X_train_raw - med) / scale).astype(np.float32, copy=False)
    X_val = ((X_val_raw - med) / scale).astype(np.float32, copy=False)
    X_test = ((X_test_raw - med) / scale).astype(np.float32, copy=False)

    out = ScaledSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )
    return out, meta


def _spec_to_row(spec: GridRunSpec, seed: int, run_id: int) -> Dict[str, object]:
    return {
        "run_id": int(run_id),
        "seed": int(seed),
        "model_type": str(spec.model_type),
        "hidden_dims": "x".join(str(v) for v in spec.hidden_dims),
        "dropout": float(spec.dropout),
        "cross_layers": int(spec.cross_layers),
        "cross_rank": int(spec.cross_rank),
        "embed_dim": int(spec.embed_dim),
        "learning_rate": float(spec.learning_rate),
        "weight_decay": float(spec.weight_decay),
        "class_weight_power": float(spec.class_weight_power),
        "label_smoothing": float(spec.label_smoothing),
        "batch_size": int(spec.batch_size),
    }


def _build_model_cfg(spec: GridRunSpec) -> ModelConfig:
    if spec.model_type == "dcn":
        return ModelConfig(
            model_type="dcn",
            hidden_dims=spec.hidden_dims,
            cross_layers=spec.cross_layers,
            cross_rank=spec.cross_rank,
            embed_dim=32,
            dropout=spec.dropout,
            activation="gelu",
        )

    if spec.model_type == "deepfm":
        return ModelConfig(
            model_type="deepfm",
            hidden_dims=spec.hidden_dims,
            cross_layers=0,
            cross_rank=0,
            embed_dim=spec.embed_dim,
            dropout=spec.dropout,
            activation="silu",
        )

    raise ValueError(f"Unsupported model_type: {spec.model_type}")


def run_broad_grid_trial(
    dataset: LoadedDataset,
    scout_cfg: ScoutConfig,
    space: GridSearchSpace | None = None,
    n_trials: int = 24,
    seeds: Sequence[int] = (42,),
    sample_seed: int = 1729,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run a broad trial sweep (sampled from large grid) on one CUDA device.

    Note:
    - Runs are sequential by design because one GPU is shared.
    - Internally metrics are vectorized and cutoff scan uses numba parallel loops.
    """
    gs_space = space or GridSearchSpace()
    pool = build_candidate_pool(gs_space)
    sampled = sample_candidates(pool, n_trials=n_trials, seed=sample_seed)

    split_data, split_meta = _make_scaled_split(dataset, scout_cfg)
    train_frame = dataset.dataframe.iloc[split_data.train_idx].copy()
    val_frame = dataset.dataframe.iloc[split_data.val_idx].copy()
    test_frame = dataset.dataframe.iloc[split_data.test_idx].copy()

    n_classes = int(np.max(dataset.y) + 1)
    rows: List[Dict[str, object]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}

    total = len(sampled) * len(seeds)
    step = 0

    for run_id, spec in enumerate(sampled, start=1):
        model_cfg = _build_model_cfg(spec)

        for seed in seeds:
            step += 1
            train_cfg = replace(
                scout_cfg.train,
                seed=int(seed),
                learning_rate=float(spec.learning_rate),
                weight_decay=float(spec.weight_decay),
                label_smoothing=float(spec.label_smoothing),
                batch_size=int(spec.batch_size),
            )

            class_weights = compute_class_weights(
                y_train=split_data.y_train,
                n_classes=n_classes,
                power=float(spec.class_weight_power),
            )

            if verbose:
                print(
                    f"[grid] {step}/{total} run_id={run_id} seed={seed} "
                    f"model={model_cfg.model_type} dims={model_cfg.hidden_dims} "
                    f"drop={model_cfg.dropout:.2f} "
                    f"cross={model_cfg.cross_layers}/{model_cfg.cross_rank} embed={model_cfg.embed_dim} "
                    f"lr={train_cfg.learning_rate:.4g} bs={train_cfg.batch_size}"
                )

            fit = fit_dcn_model(
                model_name=f"grid_trial_{model_cfg.model_type}",
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                scout_cfg=scout_cfg,
                X_train=split_data.X_train,
                y_train=split_data.y_train,
                X_val=split_data.X_val,
                y_val=split_data.y_val,
                X_test=split_data.X_test,
                y_test=split_data.y_test,
                class_weights=class_weights,
                train_frame=train_frame,
                val_frame=val_frame,
                test_frame=test_frame,
            )

            row = _spec_to_row(spec=spec, seed=int(seed), run_id=int(run_id))
            row.update(
                {
                    "cutoff_kind": str(fit.cutoff_kind),
                    "cutoff_objective": str(fit.cutoff_objective),
                    "best_epoch": int(fit.best_epoch),
                    "best_margin": float(fit.best_margin),
                    "best_val_score": float(fit.best_val_score),
                    "train_loss_last": float(fit.train_loss_last),
                }
            )
            for split_name, metrics in [
                ("train", fit.train_metrics),
                ("val", fit.val_metrics),
                ("test", fit.test_metrics),
            ]:
                for k, v in metrics.items():
                    row[f"{split_name}_{k}"] = float(v)

            rows.append(row)
            histories[f"run{run_id}_seed{seed}"] = fit.history

    results = pd.DataFrame(rows)
    if len(results):
        order: List[str] = []
        asc: List[bool] = []
        if "test_cutoff_score" in results.columns:
            order.append("test_cutoff_score")
            asc.append(False)
        for col, is_asc in [
            ("test_recall_min", False),
            ("test_recall_gap", True),
            ("test_mcc", False),
            ("test_acc", False),
        ]:
            if col in results.columns:
                order.append(col)
                asc.append(is_asc)
        results = results.sort_values(by=order, ascending=asc).reset_index(drop=True)

    meta = {
        "split_summary": split_meta,
        "n_trials_requested": int(n_trials),
        "n_trials_effective": int(len(sampled)),
        "seeds": [int(s) for s in seeds],
        "candidate_pool_size": int(len(pool)),
        "sample_seed": int(sample_seed),
        "histories": histories,
    }

    return results, meta
