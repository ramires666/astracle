"""High-level experiment runners for notebook and Python scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (
    ModelConfig,
    ScoutConfig,
    TrainConfig,
    with_dropout,
    with_embed_dim,
    with_model_dims,
    with_model_type,
    with_seed,
)
from .data_utils import LoadedDataset, build_time_split, prepare_split, split_summary
from .trainer import FitResult, fit_dcn_model


@dataclass(frozen=True)
class ScoutRunSpec:
    """Named model variant in quick scouting experiments."""

    name: str
    model: ModelConfig


def default_scout_model_grid(base: ModelConfig | None = None) -> List[ScoutRunSpec]:
    """Small but informative two-network scout grid for first checks."""
    base_cfg = base or ModelConfig()
    dcn_base = with_model_type(base_cfg, "dcn")
    deepfm_base = with_model_type(base_cfg, "deepfm")

    return [
        ScoutRunSpec(name="dcn_base", model=dcn_base),
        ScoutRunSpec(name="dcn_wider_mlp", model=with_model_dims(dcn_base, (512, 256, 128))),
        ScoutRunSpec(name="dcn_high_dropout", model=with_dropout(dcn_base, 0.30)),
        ScoutRunSpec(
            name="dcn_deep_cross",
            model=ModelConfig(
                model_type="dcn",
                hidden_dims=dcn_base.hidden_dims,
                cross_layers=5,
                cross_rank=dcn_base.cross_rank,
                embed_dim=dcn_base.embed_dim,
                dropout=dcn_base.dropout,
                activation=dcn_base.activation,
            ),
        ),
        ScoutRunSpec(name="deepfm_base", model=deepfm_base),
        ScoutRunSpec(name="deepfm_wider_mlp", model=with_model_dims(deepfm_base, (512, 256, 128))),
        ScoutRunSpec(name="deepfm_high_dropout", model=with_dropout(deepfm_base, 0.30)),
        ScoutRunSpec(name="deepfm_embed64", model=with_embed_dim(deepfm_base, 64)),
    ]


def _result_to_row(
    result: FitResult,
    seed: int,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {
        "model": result.model_name,
        "model_type": str(model_cfg.model_type),
        "seed": int(seed),
        "epochs": int(train_cfg.epochs),
        "batch_size": int(train_cfg.batch_size),
        "lr": float(train_cfg.learning_rate),
        "weight_decay": float(train_cfg.weight_decay),
        "dropout": float(model_cfg.dropout),
        "cross_layers": int(model_cfg.cross_layers),
        "cross_rank": int(model_cfg.cross_rank),
        "embed_dim": int(model_cfg.embed_dim),
        "hidden_dims": "x".join(str(v) for v in model_cfg.hidden_dims),
        "cutoff_kind": str(result.cutoff_kind),
        "best_epoch": int(result.best_epoch),
        "best_margin": float(result.best_margin),
        "best_val_score": float(result.best_val_score),
        "train_loss_last": float(result.train_loss_last),
    }

    for split_name, metrics in [
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ]:
        for key, value in metrics.items():
            row[f"{split_name}_{key}"] = float(value)

    return row


def run_quick_scout(
    dataset: LoadedDataset,
    scout_cfg: ScoutConfig,
    model_specs: Sequence[ScoutRunSpec] | None = None,
    seeds: Sequence[int] = (42, 43),
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run several short CUDA experiments to estimate model potential quickly."""
    specs = list(default_scout_model_grid() if model_specs is None else model_specs)
    if not specs:
        raise ValueError("model_specs must not be empty")

    split = build_time_split(n_rows=len(dataset.y), cfg=scout_cfg.split)
    summary = split_summary(dataset=dataset, split=split)

    n_classes = int(np.max(dataset.y) + 1)
    prepared = prepare_split(
        dataset=dataset,
        split=split,
        class_weight_power=float(scout_cfg.train.class_weight_power),
        n_classes=n_classes,
    )

    rows: List[Dict[str, float | int | str]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}

    total = len(seeds) * len(specs)
    step = 0

    for seed in seeds:
        train_cfg = with_seed(scout_cfg.train, int(seed))

        for spec in specs:
            step += 1
            if verbose:
                print(
                    f"[scout] {step}/{total} "
                    f"model={spec.name} seed={seed} "
                    f"epochs={train_cfg.epochs} bs={train_cfg.batch_size}"
                )

            fit = fit_dcn_model(
                model_name=spec.name,
                model_cfg=spec.model,
                train_cfg=train_cfg,
                scout_cfg=scout_cfg,
                X_train=prepared.X_train,
                y_train=prepared.y_train,
                X_val=prepared.X_val,
                y_val=prepared.y_val,
                X_test=prepared.X_test,
                y_test=prepared.y_test,
                class_weights=prepared.class_weights,
            )

            rows.append(
                _result_to_row(
                    result=fit,
                    seed=int(seed),
                    train_cfg=train_cfg,
                    model_cfg=spec.model,
                )
            )

            histories[f"{spec.name}__seed{seed}"] = fit.history

    results = pd.DataFrame(rows)
    if len(results):
        results = results.sort_values(
            by=["test_recall_min", "test_mcc", "test_acc"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    meta = {
        "split_summary": summary,
        "histories": histories,
        "n_runs": int(total),
        "n_classes": int(n_classes),
    }

    return results, meta


def suggest_tuning_bounds(results: pd.DataFrame, top_k: int = 3) -> Dict[str, object]:
    """Build rough variable bounds from the best scout runs."""
    if results.empty:
        return {}

    top = results.head(int(max(top_k, 1))).copy()
    widths = top["hidden_dims"].astype(str).tolist()

    return {
        "recommended_model_types": sorted(set(top["model_type"].astype(str).tolist())),
        "recommended_margin_min": float(max(0.0, top["best_margin"].min() * 0.5)),
        "recommended_margin_max": float(top["best_margin"].max() * 1.8 + 1e-9),
        "recommended_dropout_min": float(max(0.05, top["dropout"].min() - 0.05)),
        "recommended_dropout_max": float(min(0.50, top["dropout"].max() + 0.10)),
        "recommended_cross_layers": sorted({int(v) for v in top["cross_layers"].tolist()}),
        "recommended_cross_rank": sorted({int(v) for v in top["cross_rank"].tolist()}),
        "recommended_hidden_dims_examples": widths,
        "best_test_recall_min": float(top["test_recall_min"].max()),
        "best_test_mcc": float(top["test_mcc"].max()),
    }


def save_scout_results(results: pd.DataFrame, out_csv: str | None) -> None:
    """Save scout table if path is provided."""
    if out_csv is None:
        return
    results.to_csv(out_csv, index=False)
