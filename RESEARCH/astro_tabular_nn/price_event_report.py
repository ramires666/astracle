"""CLI: build price-vs-truth alignment charts for selected scout models."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from RESEARCH.data_loader import load_market_data

from .best_grid_dataset import ensure_best_grid_dataset_path
from .config import DatasetConfig, ModelConfig, ScoutConfig, TrainConfig, with_dataset
from .data_utils import build_time_split, load_tabular_dataset, prepare_split
from .event_alignment import (
    build_global_true_events,
    build_pred_regime_labels_from_frame,
    build_switch_events,
    compute_event_metrics,
    match_events_by_window,
    plot_event_alignment,
)
from .trainer import fit_dcn_model


def _parse_hidden_dims(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    text = str(value).strip().lower()
    if not text:
        raise ValueError("hidden_dims is empty")
    parts = [p for p in text.replace(",", "x").split("x") if p.strip()]
    return tuple(int(p) for p in parts)


def _find_latest_selected_models_csv() -> Path:
    base = Path("RESEARCH/reports")
    candidates = sorted(
        base.glob("astro_tabular_nn_postrun_wide_balance_*/selected_models.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No selected_models.csv found in latest postrun folders")
    return candidates[-1]


def _resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset:
        return Path(args.dataset)
    if args.dataset_source == "best-grid":
        return ensure_best_grid_dataset_path(
            run_tag=str(args.run_tag),
            data_start=str(args.data_start),
            use_cache=True,
            verbose=True,
        )
    return DatasetConfig().dataset_path


def _threshold_grid(min_v: float, max_v: float, step: float) -> Tuple[float, ...]:
    values = np.arange(float(min_v), float(max_v) + 1e-12, float(step), dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    values = np.unique(np.round(values, 6))
    return tuple(float(v) for v in values.tolist())


def _build_model_cfg(row: pd.Series) -> ModelConfig:
    model_type = str(row["model_type"]).lower().strip()
    hidden_dims = _parse_hidden_dims(row["hidden_dims"])
    activation = "gelu" if model_type == "dcn" else "silu"
    return ModelConfig(
        model_type=model_type,
        hidden_dims=hidden_dims,
        cross_layers=int(row.get("cross_layers", 0)),
        cross_rank=int(row.get("cross_rank", 0)),
        embed_dim=int(row.get("embed_dim", 32)),
        dropout=float(row["dropout"]),
        activation=activation,
    )


def _build_train_cfg(
    row: pd.Series,
    epochs_override: Optional[int],
    class_weight_power: float,
    label_smoothing: float,
    early_stopping_patience: int,
) -> TrainConfig:
    base = TrainConfig()
    epochs = int(epochs_override) if epochs_override is not None else int(row["epochs"])
    return replace(
        base,
        seed=int(row["seed"]),
        epochs=epochs,
        batch_size=int(row["batch_size"]),
        learning_rate=float(row["lr"]),
        weight_decay=float(row["weight_decay"]),
        class_weight_power=float(class_weight_power),
        label_smoothing=float(label_smoothing),
        early_stopping_patience=int(early_stopping_patience),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build old-notebook-style price alignment charts for selected models")
    p.add_argument("--selected-csv", type=str, default=None, help="Path to selected_models.csv (default: latest)")
    p.add_argument("--dataset", type=str, default=None, help="Optional parquet dataset path")
    p.add_argument("--dataset-source", type=str, choices=["best-grid", "default-parquet"], default="best-grid")
    p.add_argument("--run-tag", type=str, default="turning_massive_label_grid")
    p.add_argument("--data-start", type=str, default="2017-11-01")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--window-days", type=int, default=10)
    p.add_argument("--true-mode", type=str, choices=["global_turning_points", "target_switch"], default="global_turning_points")
    p.add_argument("--pred-mode", type=str, choices=["hard_label", "proba_smooth_regime", "proba_threshold_centered"], default="hard_label")
    p.add_argument("--pred-smooth-span-days", type=int, default=9)
    p.add_argument("--pred-enter-up", type=float, default=0.55)
    p.add_argument("--pred-enter-down", type=float, default=0.45)
    p.add_argument("--pred-min-segment-days", type=int, default=10)
    p.add_argument("--center-threshold", type=float, default=0.50)
    p.add_argument("--center-delta-up", type=float, default=0.03)
    p.add_argument("--center-delta-down", type=float, default=0.03)
    p.add_argument("--epochs-override", type=int, default=None)
    p.add_argument("--class-weight-power", type=float, default=1.4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--gap-penalty", type=float, default=0.70)
    p.add_argument("--prior-penalty", type=float, default=0.12)
    p.add_argument("--threshold-min", type=float, default=0.10)
    p.add_argument("--threshold-max", type=float, default=0.90)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--tp-horizon-days", type=int, default=10)
    p.add_argument("--tp-up-move-pct", type=float, default=0.09)
    p.add_argument("--tp-down-move-pct", type=float, default=0.09)
    p.add_argument("--tp-cluster-gap-days", type=int, default=10)
    p.add_argument("--tp-min-turn-gap-days", type=int, default=14)
    p.add_argument("--tp-past-horizon-days", type=int, default=10)
    p.add_argument("--tp-past-up-move-pct", type=float, default=0.09)
    p.add_argument("--tp-past-down-move-pct", type=float, default=0.09)
    p.add_argument("--tp-tail-direction-mode", type=str, default="endpoint_sign")
    p.add_argument("--tp-tail-min-move-pct", type=float, default=0.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    selected_csv = Path(args.selected_csv) if args.selected_csv else _find_latest_selected_models_csv()
    selected = pd.read_csv(selected_csv)
    if "global_rank" in selected.columns:
        selected = selected.sort_values("global_rank", ascending=True)
    selected = selected.head(int(max(1, args.top_k))).reset_index(drop=True)

    dataset_path = _resolve_dataset_path(args)
    dataset = load_tabular_dataset(with_dataset(DatasetConfig(), dataset_path))
    split = build_time_split(n_rows=len(dataset.y), cfg=ScoutConfig().split)
    n_classes = int(np.max(dataset.y) + 1)
    prepared = prepare_split(
        dataset=dataset,
        split=split,
        class_weight_power=float(args.class_weight_power),
        n_classes=n_classes,
    )

    test_rows = dataset.dataframe.iloc[split.test_idx].copy().reset_index(drop=True)
    if "close" not in test_rows.columns:
        raise KeyError("Dataset must contain 'close' column for price chart")

    full_true_events: Optional[pd.DataFrame] = None
    if args.true_mode == "global_turning_points":
        df_market = load_market_data(start_date=str(args.data_start))
        df_market = df_market[["date", "close"]].copy()
        df_market["date"] = pd.to_datetime(df_market["date"])
        df_market = df_market.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        full_true_events = build_global_true_events(
            df_market_ref=df_market,
            horizon_days=int(args.tp_horizon_days),
            up_move_pct=float(args.tp_up_move_pct),
            down_move_pct=float(args.tp_down_move_pct),
            cluster_gap_days=int(args.tp_cluster_gap_days),
            min_turn_gap_days=int(args.tp_min_turn_gap_days),
            past_horizon_days=int(args.tp_past_horizon_days),
            past_up_move_pct=float(args.tp_past_up_move_pct),
            past_down_move_pct=float(args.tp_past_down_move_pct),
            tail_direction_mode=str(args.tp_tail_direction_mode),
            tail_min_move_pct=float(args.tp_tail_min_move_pct),
        )

    threshold_grid = _threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"RESEARCH/reports/astro_tabular_nn_price_event_report_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for idx, row in selected.iterrows():
        rank = int(row["global_rank"]) if "global_rank" in row else int(idx + 1)
        model_name = str(row["model"])

        train_cfg = _build_train_cfg(
            row=row,
            epochs_override=args.epochs_override,
            class_weight_power=args.class_weight_power,
            label_smoothing=args.label_smoothing,
            early_stopping_patience=args.early_stopping_patience,
        )
        scout_cfg = replace(
            ScoutConfig(train=train_cfg),
            threshold_grid=threshold_grid,
            margin_gap_penalty=float(args.gap_penalty),
            margin_prior_penalty=float(args.prior_penalty),
        )

        fit = fit_dcn_model(
            model_name=f"price_event_{model_name}",
            model_cfg=_build_model_cfg(row),
            train_cfg=train_cfg,
            scout_cfg=scout_cfg,
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
            class_weights=prepared.class_weights,
            capture_predictions=True,
        )
        if fit.test_pred is None or fit.test_proba is None:
            raise RuntimeError("Missing captured predictions from trainer")

        test_frame = test_rows[["date", "close"]].copy()
        test_frame["target"] = prepared.y_test.astype(np.int32)
        test_frame["pred"] = fit.test_pred.astype(np.int32)
        test_frame["proba_up"] = fit.test_proba[:, 1].astype(float) if fit.test_proba.shape[1] > 1 else np.full(len(test_frame), np.nan)

        if args.true_mode == "target_switch":
            true_events = build_switch_events(test_frame["date"], test_frame["target"].to_numpy(dtype=np.int32))
        else:
            if full_true_events is None:
                raise RuntimeError("Global true events are not initialized")
            d0 = pd.to_datetime(test_frame["date"].min())
            d1 = pd.to_datetime(test_frame["date"].max())
            true_events = full_true_events[(full_true_events["date"] >= d0) & (full_true_events["date"] <= d1)].copy().reset_index(drop=True)

        pred_labels = build_pred_regime_labels_from_frame(
            test_frame=test_frame,
            mode=args.pred_mode,
            smooth_span_days=args.pred_smooth_span_days,
            enter_up=args.pred_enter_up,
            enter_down=args.pred_enter_down,
            min_segment_days=args.pred_min_segment_days,
            center_threshold=args.center_threshold,
            center_delta_up=args.center_delta_up,
            center_delta_down=args.center_delta_down,
        )
        pred_events = build_switch_events(test_frame["date"], pred_labels)
        matches = match_events_by_window(true_events=true_events, pred_events=pred_events, window_days=args.window_days)
        ev = compute_event_metrics(true_events=true_events, pred_events=pred_events, matches=matches)

        stem = f"rank{rank:02d}_{model_name}_seed{int(row['seed'])}"
        plot_path = out_dir / f"{stem}_price_event_alignment.png"
        plot_event_alignment(
            test_frame=test_frame,
            true_events=true_events,
            pred_labels=pred_labels,
            matches=matches,
            title_prefix=stem,
            window_days=args.window_days,
            true_mode=args.true_mode,
            out_png=plot_path,
        )

        test_frame.to_csv(out_dir / f"{stem}_test_frame.csv", index=False)
        true_events.to_csv(out_dir / f"{stem}_true_events.csv", index=False)
        pred_events.to_csv(out_dir / f"{stem}_pred_events.csv", index=False)
        matches.to_csv(out_dir / f"{stem}_event_matches.csv", index=False)

        summary_rows.append(
            {
                "global_rank": rank,
                "model": model_name,
                "model_type": str(row["model_type"]),
                "seed": int(row["seed"]),
                "test_recall_min": float(row["test_recall_min"]),
                "test_recall_gap": float(row["test_recall_gap"]),
                "test_mcc": float(row["test_mcc"]),
                "test_acc": float(row["test_acc"]),
                "event_recall_true": float(ev["recall_true"]),
                "event_precision_pred": float(ev["precision_pred"]),
                "event_mean_abs_lag_days": float(ev["mean_abs_lag_days"]),
                "n_true_events": int(ev["n_true_events"]),
                "n_pred_events": int(ev["n_pred_events"]),
                "n_matched_events": int(ev["n_matched"]),
                "plot_path": str(plot_path),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["global_rank", "event_recall_true"], ascending=[True, False])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# Price Event Alignment Report",
        "",
        f"- Selected CSV: `{selected_csv}`",
        f"- Dataset path: `{dataset_path}`",
        f"- True mode: `{args.true_mode}`",
        f"- Pred mode: `{args.pred_mode}`",
        f"- Event window: `±{args.window_days} days`",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[price_event_report] selected_csv={selected_csv}")
    print(f"[price_event_report] dataset={dataset_path}")
    print(f"[price_event_report] out_dir={out_dir}")
    print(f"[price_event_report] summary={summary_path}")


if __name__ == "__main__":
    main()

