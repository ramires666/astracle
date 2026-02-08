"""
Train production artifacts from the statistically analyzed turning-grid candidate.

Default behavior:
- Select best candidate from TP segment-weighted stats.
- Rebuild turning dataset with that candidate params.
- Train split artifact (same 70/15/15 setup as research).
- Optionally train full artifact for forecast endpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from production_dev.turning_pipeline import (
    build_artifact_config_from_checkpoint_row,
    build_turning_dataset_from_artifact_config,
    load_market_frame,
    train_turning_full_model,
    train_turning_split_model,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "market" / "reports" / "turning_massive_label_grid_checkpoint.csv"
DEFAULT_SEGMENT_STATS = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "reports"
    / "turning_massive_label_grid_top100_tp_segment_weighted_g1p5_mind5_tail1_pred-hard_label.csv"
)
DEFAULT_SPLIT_OUT = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.turning_split.joblib"
DEFAULT_FULL_OUT = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.turning_full.joblib"


def _sort_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same ranking logic as turning-grid notebook.
    """
    out = df.copy()
    defaults = {
        "is_feasible": 0,
        "total_constraint_violation": 1e9,
        "test_profit_y_obj": -1e9,
        "test_profit_y": -1e9,
        "test_recall_min": -1.0,
        "test_recall_gap": 1e9,
        "mcc": -1e9,
    }
    for key, val in defaults.items():
        if key not in out.columns:
            out[key] = val
    return out.sort_values(
        [
            "is_feasible",
            "total_constraint_violation",
            "test_profit_y_obj",
            "test_profit_y",
            "test_recall_min",
            "test_recall_gap",
            "mcc",
        ],
        ascending=[False, True, False, False, False, True, False],
    ).reset_index(drop=True)


def _pick_candidate_row(
    df_checkpoint: pd.DataFrame,
    strategy: str,
    segment_stats_path: Path,
    eval_id: int | None,
) -> pd.Series:
    """
    Pick one checkpoint row by requested strategy.
    """
    if eval_id is not None:
        row = df_checkpoint[df_checkpoint["eval_id"] == int(eval_id)]
        if row.empty:
            raise ValueError(f"eval_id={eval_id} not found in checkpoint.")
        return row.iloc[0].copy()

    if strategy == "segment_best":
        if not segment_stats_path.exists():
            raise FileNotFoundError(
                f"segment stats file not found: {segment_stats_path}. "
                "Use --strategy grid_best or provide --eval-id."
            )
        df_seg = pd.read_csv(segment_stats_path)
        if df_seg.empty:
            raise ValueError(f"segment stats file is empty: {segment_stats_path}")
        df_seg = df_seg.sort_values(
            ["p_shift_weighted_hit", "delta_weighted_hit_vs_null", "test_profit_y_obj"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        best_eval = int(df_seg.iloc[0]["eval_id"])
        row = df_checkpoint[df_checkpoint["eval_id"] == best_eval]
        if row.empty:
            raise ValueError(f"segment-best eval_id={best_eval} not found in checkpoint.")
        return row.iloc[0].copy()

    if strategy == "grid_best":
        return _sort_results_frame(df_checkpoint).iloc[0].copy()

    raise ValueError(f"Unsupported strategy: {strategy}")


def _build_artifact_payload(
    model: Any,
    feature_names: list[str],
    config: dict[str, Any],
    extra_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Build common artifact dictionary.
    """
    return {
        "model": model,
        "feature_names": feature_names,
        "config": config,
        **extra_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train turning-grid production artifacts from checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--segment-stats", type=Path, default=DEFAULT_SEGMENT_STATS)
    parser.add_argument(
        "--strategy",
        type=str,
        default="segment_best",
        choices=["segment_best", "grid_best"],
        help="Candidate selection strategy when --eval-id is not provided.",
    )
    parser.add_argument("--eval-id", type=int, default=None, help="Force exact eval_id from checkpoint.")
    parser.add_argument("--data-start", type=str, default="2017-11-01")
    parser.add_argument("--split-output", type=Path, default=DEFAULT_SPLIT_OUT)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUT)
    parser.add_argument("--skip-full", action="store_true", help="Train only split artifact.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    df_checkpoint = pd.read_csv(args.checkpoint)
    if df_checkpoint.empty:
        raise ValueError(f"checkpoint is empty: {args.checkpoint}")

    row = _pick_candidate_row(
        df_checkpoint=df_checkpoint,
        strategy=str(args.strategy),
        segment_stats_path=args.segment_stats,
        eval_id=args.eval_id,
    )

    artifact_cfg = build_artifact_config_from_checkpoint_row(row)
    print("=" * 90)
    print("Selected turning-grid candidate")
    print(
        f"eval_id={int(row.get('eval_id', -1))} "
        f"threshold={float(artifact_cfg.get('decision_threshold', 0.5)):.3f} "
        f"target_mode={artifact_cfg['target_cfg']['mode']}"
    )
    print(
        f"test_profit_y_obj={float(row.get('test_profit_y_obj', np.nan)):+.8f} "
        f"test_recall_min={float(row.get('test_recall_min', np.nan)):.3f} "
        f"test_recall_gap={float(row.get('test_recall_gap', np.nan)):.3f} "
        f"mcc={float(row.get('mcc', np.nan)):.3f}"
    )
    print("=" * 90)

    df_market = load_market_frame(data_start=str(args.data_start))
    print(
        f"Market rows={len(df_market)} "
        f"range={df_market['date'].min().date()}..{df_market['date'].max().date()}"
    )

    df_dataset, feature_cols = build_turning_dataset_from_artifact_config(
        df_market=df_market,
        config=artifact_cfg,
        cache_namespace="research2_turning_grid",
        use_cache=True,
        verbose=True,
    )
    print(f"Turning dataset rows={len(df_dataset)} features={len(feature_cols)}")

    split_res = train_turning_split_model(
        df_dataset=df_dataset,
        feature_cols=feature_cols,
        config=artifact_cfg,
        seed=int(args.seed),
    )

    # Keep UI-compatible keys in config.
    artifact_cfg["n_estimators"] = int(artifact_cfg["model_cfg"]["n_estimators"])
    artifact_cfg["max_depth"] = int(artifact_cfg["model_cfg"]["max_depth"])
    artifact_cfg["learning_rate"] = float(artifact_cfg["model_cfg"]["learning_rate"])
    artifact_cfg["subsample"] = float(artifact_cfg["model_cfg"]["subsample"])
    artifact_cfg["colsample_bytree"] = float(artifact_cfg["model_cfg"]["colsample_bytree"])
    artifact_cfg["r_min"] = float(split_res.test_metrics["recall_min"])
    artifact_cfg["mcc"] = float(split_res.test_metrics["mcc"])

    split_meta = {
        "source_checkpoint": str(args.checkpoint),
        "source_eval_id": int(row.get("eval_id", -1)),
        "source_candidate_rank": int(row.get("candidate_rank", -1)) if np.isfinite(pd.to_numeric(row.get("candidate_rank", np.nan), errors="coerce")) else -1,
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "train_samples": int(split_res.train_rows),
        "val_samples": int(split_res.val_rows),
        "test_samples": int(split_res.test_rows),
        "decision_threshold": float(split_res.threshold),
        "val_metrics": split_res.val_metrics,
        "test_metrics": split_res.test_metrics,
        "val_profit_y": float(split_res.val_profit_y),
        "test_profit_y": float(split_res.test_profit_y),
        "val_profit_y_obj": float(split_res.val_profit_y_obj),
        "test_profit_y_obj": float(split_res.test_profit_y_obj),
    }

    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    split_payload = _build_artifact_payload(
        model=split_res.model,
        feature_names=split_res.feature_names,
        config=artifact_cfg,
        extra_meta=split_meta,
    )
    joblib.dump(split_payload, args.split_output)
    print(f"✅ Saved split artifact: {args.split_output}")
    print(
        "   test:",
        f"r_min={split_res.test_metrics['recall_min']:.3f}",
        f"gap={split_res.test_metrics['recall_gap']:.3f}",
        f"mcc={split_res.test_metrics['mcc']:.3f}",
        f"profit_obj={split_res.test_profit_y_obj:+.8f}",
    )

    if not args.skip_full:
        full_model = train_turning_full_model(
            df_dataset=df_dataset,
            feature_cols=feature_cols,
            config=artifact_cfg,
            seed=int(args.seed),
        )
        full_meta = {
            "source_checkpoint": str(args.checkpoint),
            "source_eval_id": int(row.get("eval_id", -1)),
            "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "train_samples": int(len(df_dataset)),
            "decision_threshold": float(artifact_cfg["decision_threshold"]),
            "is_full_model": True,
        }
        args.full_output.parent.mkdir(parents=True, exist_ok=True)
        full_payload = _build_artifact_payload(
            model=full_model,
            feature_names=feature_cols,
            config=artifact_cfg,
            extra_meta=full_meta,
        )
        joblib.dump(full_payload, args.full_output)
        print(f"✅ Saved full artifact:  {args.full_output}")


if __name__ == "__main__":
    main()
