"""Build notebook-style markdown report with embedded charts for winner-grid runs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _find_latest(path_glob: str) -> Path:
    items = sorted(Path("RESEARCH/reports").glob(path_glob), key=lambda p: p.stat().st_mtime)
    if not items:
        raise FileNotFoundError(f"No files matched: RESEARCH/reports/{path_glob}")
    return items[-1]


def _sorted_results(df: pd.DataFrame) -> pd.DataFrame:
    if "test_cutoff_score" in df.columns:
        return df.sort_values(
            ["test_cutoff_score", "test_recall_min", "test_recall_gap", "test_mcc", "test_acc"],
            ascending=[False, False, True, False, False],
        ).reset_index(drop=True)
    return df.sort_values(
        ["test_recall_min", "test_recall_gap", "test_mcc", "test_acc"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def _f4(value: float) -> str:
    return f"{float(value):.4f}"


def _copy_if_exists(src: Path, out_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _plot_recall_vs_gap(df: pd.DataFrame, baseline: pd.Series, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    x = pd.to_numeric(df["test_recall_gap"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["test_recall_min"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(df["test_mcc"], errors="coerce").to_numpy(dtype=float)
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=36, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="test_mcc")

    best = _sorted_results(df).iloc[0]
    ax.scatter(
        [float(best["test_recall_gap"])],
        [float(best["test_recall_min"])],
        marker="*",
        s=260,
        color="black",
        label="best (new grid)",
        zorder=6,
    )
    ax.scatter(
        [float(baseline["test_recall_gap"])],
        [float(baseline["test_recall_min"])],
        marker="X",
        s=180,
        color="#d62728",
        label="baseline winner",
        zorder=6,
    )
    ax.set_xlabel("test_recall_gap (lower is better)")
    ax.set_ylabel("test_recall_min (higher is better)")
    ax.set_title("Winner Grid: Recall-Min vs Recall-Gap")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_distributions(df: pd.DataFrame, baseline: pd.Series, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    rec = pd.to_numeric(df["test_recall_min"], errors="coerce").to_numpy(dtype=float)
    gap = pd.to_numeric(df["test_recall_gap"], errors="coerce").to_numpy(dtype=float)

    axes[0].hist(rec[np.isfinite(rec)], bins=20, color="#1f77b4", alpha=0.85)
    axes[0].axvline(float(baseline["test_recall_min"]), color="#d62728", linestyle="--", linewidth=1.8, label="baseline")
    axes[0].set_title("Distribution: test_recall_min")
    axes[0].set_xlabel("test_recall_min")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.25, linestyle=":")
    axes[0].legend(loc="best")

    axes[1].hist(gap[np.isfinite(gap)], bins=20, color="#ff7f0e", alpha=0.85)
    axes[1].axvline(float(baseline["test_recall_gap"]), color="#d62728", linestyle="--", linewidth=1.8, label="baseline")
    axes[1].set_title("Distribution: test_recall_gap")
    axes[1].set_xlabel("test_recall_gap")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.25, linestyle=":")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_top_recall_profiles(top_df: pd.DataFrame, out_png: Path, top_n: int = 10) -> None:
    top = top_df.head(int(max(1, top_n))).copy()
    labels = [f"#{i + 1}" for i in range(len(top))]
    x = np.arange(len(top), dtype=float)
    w = 0.28

    down = pd.to_numeric(top["test_recall_down"], errors="coerce").to_numpy(dtype=float)
    up = pd.to_numeric(top["test_recall_up"], errors="coerce").to_numpy(dtype=float)
    rmin = pd.to_numeric(top["test_recall_min"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.bar(x - 0.5 * w, down, width=w, color="#377eb8", label="recall_down")
    ax.bar(x + 0.5 * w, up, width=w, color="#4daf4a", label="recall_up")
    ax.plot(x, rmin, color="#222222", marker="o", linewidth=1.5, label="recall_min")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Top-ranked configurations")
    ax.set_ylabel("Recall")
    ax.set_title("Top Configs: Recall Profiles")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_vs_best(best: pd.Series, baseline: pd.Series, out_png: Path) -> None:
    metric_names = ["test_recall_min", "test_recall_gap", "test_mcc", "test_acc"]
    labels = ["recall_min", "recall_gap", "mcc", "acc"]
    b_vals = [float(baseline[k]) for k in metric_names]
    n_vals = [float(best[k]) for k in metric_names]
    x = np.arange(len(labels), dtype=float)
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.bar(x - 0.5 * w, b_vals, width=w, color="#d62728", alpha=0.88, label="baseline")
    ax.bar(x + 0.5 * w, n_vals, width=w, color="#1f77b4", alpha=0.88, label="new_best")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Baseline vs New Best")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build winner-grid markdown report with embedded charts")
    p.add_argument("--grid-csv", type=str, default=None, help="Grid CSV path (default: latest winner grid csv)")
    p.add_argument("--top15-csv", type=str, default=None, help="Optional top15 CSV path")
    p.add_argument("--meta-json", type=str, default=None, help="Optional meta JSON path")
    p.add_argument(
        "--baseline-selected-csv",
        type=str,
        default="RESEARCH/reports/astro_tabular_nn_postrun_wide_balance_20260208_055915/selected_models.csv",
        help="Path to selected_models.csv used as baseline reference",
    )
    p.add_argument("--dataset", type=str, default=None, help="Optional dataset parquet path for best-run replay")
    p.add_argument("--run-tag", type=str, default="turning_massive_label_grid", help="Dataset run tag fallback")
    p.add_argument("--data-start", type=str, default="2017-11-01", help="Market start date for global TP events")
    p.add_argument("--replay-epochs", type=int, default=8, help="Epochs for replaying best run for price chart")
    p.add_argument("--replay-patience", type=int, default=5, help="Early stopping patience for replay")
    p.add_argument("--window-days", type=int, default=10, help="Event matching window for price chart")
    p.add_argument("--gap-penalty", type=float, default=0.70, help="Threshold search gap penalty")
    p.add_argument("--prior-penalty", type=float, default=0.12, help="Threshold search prior penalty")
    p.add_argument(
        "--cutoff-objective",
        type=str,
        choices=["auto", "recall_balance", "segment_weighted"],
        default="auto",
        help="Cutoff objective for best-run replay; auto takes value from grid row",
    )
    p.add_argument("--segment-score-gamma", type=float, default=1.5, help="TP segment amplitude exponent gamma")
    p.add_argument("--segment-min-days", type=int, default=5, help="Minimum TP segment days")
    p.add_argument(
        "--segment-no-open-tail",
        action="store_true",
        help="Disable open-tail segment in TP segment objective for replay",
    )
    p.add_argument("--threshold-min", type=float, default=0.10, help="Threshold grid min for replay")
    p.add_argument("--threshold-max", type=float, default=0.90, help="Threshold grid max for replay")
    p.add_argument("--threshold-step", type=float, default=0.01, help="Threshold grid step for replay")
    p.add_argument("--out-dir", type=str, default=None, help="Output report directory")
    p.add_argument("--top-k", type=int, default=15, help="Top rows to show in markdown table")
    return p


def _resolve_default_grid_csv() -> Path:
    return _find_latest("astro_tabular_nn_grid_winner_dcn_balance_*.csv")


def _resolve_sidecar(path: Path, suffix: str) -> Path:
    side = path.with_suffix(suffix)
    if side.exists():
        return side
    name = path.name
    if name.endswith(".csv"):
        alt = path.parent / name.replace(".csv", suffix)
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Sidecar file not found for {path}: {suffix}")


def _leaderboard_cols() -> Sequence[str]:
    return [
        "run_id",
        "seed",
        "model_type",
        "hidden_dims",
        "dropout",
        "cross_layers",
        "cross_rank",
        "learning_rate",
        "weight_decay",
        "class_weight_power",
        "label_smoothing",
        "batch_size",
        "cutoff_objective",
        "best_margin",
        "best_val_score",
        "test_cutoff_score",
        "test_segment_weighted_hit_rate",
        "test_segment_weighted_majority_hit",
        "test_recall_down",
        "test_recall_up",
        "test_recall_min",
        "test_recall_gap",
        "test_mcc",
        "test_acc",
    ]


def _parse_hidden_dims(value: object) -> tuple[int, ...]:
    text = str(value).strip().lower()
    parts = [p for p in text.replace(",", "x").split("x") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid hidden_dims value: {value}")
    return tuple(int(p) for p in parts)


def _threshold_grid(min_v: float, max_v: float, step: float) -> tuple[float, ...]:
    values = np.arange(float(min_v), float(max_v) + 1e-12, float(step), dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    values = np.unique(np.round(values, 6))
    return tuple(float(v) for v in values.tolist())


def _resolve_dataset_path(args: argparse.Namespace, meta_json: Path) -> Path:
    if args.dataset is not None:
        return Path(args.dataset)
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            ds = meta.get("dataset_path")
            if ds:
                p = Path(str(ds))
                if p.exists():
                    return p
        except Exception:
            pass
    return ensure_best_grid_dataset_path(
        run_tag=str(args.run_tag),
        data_start=str(args.data_start),
        use_cache=True,
        verbose=True,
    )


def _replay_best_and_plot_price_alignment(
    best: pd.Series,
    dataset_path: Path,
    out_dir: Path,
    replay_epochs: int,
    replay_patience: int,
    gap_penalty: float,
    prior_penalty: float,
    cutoff_objective: str,
    segment_score_gamma: float,
    segment_min_days: int,
    segment_include_open_tail: bool,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    data_start: str,
    window_days: int,
) -> dict[str, float | int | str]:
    dataset = load_tabular_dataset(with_dataset(DatasetConfig(), dataset_path))
    split = build_time_split(n_rows=len(dataset.y), cfg=ScoutConfig().split)
    n_classes = int(np.max(dataset.y) + 1)

    prepared = prepare_split(
        dataset=dataset,
        split=split,
        class_weight_power=float(best["class_weight_power"]),
        n_classes=n_classes,
    )

    model_type = str(best["model_type"]).lower().strip()
    model_cfg = ModelConfig(
        model_type=model_type,
        hidden_dims=_parse_hidden_dims(best["hidden_dims"]),
        cross_layers=int(best.get("cross_layers", 0)),
        cross_rank=int(best.get("cross_rank", 0)),
        embed_dim=int(best.get("embed_dim", 32)),
        dropout=float(best["dropout"]),
        activation="gelu" if model_type == "dcn" else "silu",
    )
    train_cfg = replace(
        TrainConfig(),
        seed=int(best["seed"]),
        epochs=int(replay_epochs),
        batch_size=int(best["batch_size"]),
        learning_rate=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
        class_weight_power=float(best["class_weight_power"]),
        label_smoothing=float(best["label_smoothing"]),
        early_stopping_patience=int(replay_patience),
    )
    objective = str(cutoff_objective).strip().lower()
    if objective == "auto":
        objective = str(best.get("cutoff_objective", "recall_balance")).strip().lower()

    scout_cfg = replace(
        ScoutConfig(train=train_cfg),
        threshold_grid=_threshold_grid(threshold_min, threshold_max, threshold_step),
        margin_gap_penalty=float(gap_penalty),
        margin_prior_penalty=float(prior_penalty),
        cutoff_objective=objective,
        segment_score_gamma=float(segment_score_gamma),
        segment_min_days=int(segment_min_days),
        segment_include_open_tail=bool(segment_include_open_tail),
    )

    fit = fit_dcn_model(
        model_name="winner_grid_report_best_replay",
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        scout_cfg=scout_cfg,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_val=prepared.X_val,
        y_val=prepared.y_val,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        class_weights=prepared.class_weights,
        train_frame=dataset.dataframe.iloc[split.train_idx].copy(),
        val_frame=dataset.dataframe.iloc[split.val_idx].copy(),
        test_frame=dataset.dataframe.iloc[split.test_idx].copy(),
        capture_predictions=True,
    )
    if fit.test_pred is None or fit.test_proba is None:
        raise RuntimeError("Replay fit did not return captured predictions")

    test_frame = dataset.dataframe.iloc[split.test_idx][["date", "close"]].copy().reset_index(drop=True)
    test_frame["target"] = prepared.y_test.astype(np.int32)
    test_frame["pred"] = fit.test_pred.astype(np.int32)
    test_frame["proba_up"] = (
        fit.test_proba[:, 1].astype(float) if fit.test_proba.shape[1] > 1 else np.full(len(test_frame), np.nan)
    )

    df_market = load_market_data(start_date=str(data_start))
    df_market = df_market[["date", "close"]].copy()
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    true_events_all = build_global_true_events(
        df_market_ref=df_market,
        horizon_days=10,
        up_move_pct=0.09,
        down_move_pct=0.09,
        cluster_gap_days=10,
        min_turn_gap_days=14,
        past_horizon_days=10,
        past_up_move_pct=0.09,
        past_down_move_pct=0.09,
        tail_direction_mode="endpoint_sign",
        tail_min_move_pct=0.0,
    )

    d0 = pd.to_datetime(test_frame["date"].min())
    d1 = pd.to_datetime(test_frame["date"].max())
    true_events = true_events_all[(true_events_all["date"] >= d0) & (true_events_all["date"] <= d1)].copy().reset_index(drop=True)
    pred_labels = build_pred_regime_labels_from_frame(
        test_frame=test_frame,
        mode="hard_label",
        smooth_span_days=9,
        enter_up=0.55,
        enter_down=0.45,
        min_segment_days=10,
        center_threshold=0.50,
        center_delta_up=0.03,
        center_delta_down=0.03,
    )
    pred_events = build_switch_events(test_frame["date"], pred_labels)
    matches = match_events_by_window(true_events=true_events, pred_events=pred_events, window_days=int(window_days))
    ev = compute_event_metrics(true_events=true_events, pred_events=pred_events, matches=matches)

    png_path = out_dir / "chart_price_event_alignment_best.png"
    plot_event_alignment(
        test_frame=test_frame,
        true_events=true_events,
        pred_labels=pred_labels,
        matches=matches,
        title_prefix="winner_best_replay",
        window_days=int(window_days),
        true_mode="global_turning_points",
        out_png=png_path,
    )

    test_frame.to_csv(out_dir / "best_price_event_test_frame.csv", index=False)
    true_events.to_csv(out_dir / "best_price_event_true_events.csv", index=False)
    pred_events.to_csv(out_dir / "best_price_event_pred_events.csv", index=False)
    matches.to_csv(out_dir / "best_price_event_matches.csv", index=False)

    return {
        "plot_path": str(png_path),
        "event_recall_true": float(ev["recall_true"]),
        "event_precision_pred": float(ev["precision_pred"]),
        "event_mean_abs_lag_days": float(ev["mean_abs_lag_days"]),
        "n_true_events": int(ev["n_true_events"]),
        "n_pred_events": int(ev["n_pred_events"]),
        "n_matched_events": int(ev["n_matched"]),
    }


def main() -> None:
    args = _build_parser().parse_args()
    grid_csv = Path(args.grid_csv) if args.grid_csv else _resolve_default_grid_csv()
    top15_csv = Path(args.top15_csv) if args.top15_csv else _resolve_sidecar(grid_csv, ".top15.csv")
    meta_json = Path(args.meta_json) if args.meta_json else _resolve_sidecar(grid_csv, ".meta.json")
    baseline_csv = Path(args.baseline_selected_csv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"RESEARCH/reports/astro_tabular_nn_winner_grid_report_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _sorted_results(pd.read_csv(grid_csv))
    baseline_row = pd.read_csv(baseline_csv).iloc[0]
    top = df.head(int(max(1, args.top_k))).copy()
    best = df.iloc[0]

    copied_grid = _copy_if_exists(grid_csv, out_dir)
    copied_top15 = _copy_if_exists(top15_csv, out_dir)
    copied_meta = _copy_if_exists(meta_json, out_dir)

    dataset_path = _resolve_dataset_path(args=args, meta_json=meta_json)
    price_ev = _replay_best_and_plot_price_alignment(
        best=best,
        dataset_path=dataset_path,
        out_dir=out_dir,
        replay_epochs=int(args.replay_epochs),
        replay_patience=int(args.replay_patience),
        gap_penalty=float(args.gap_penalty),
        prior_penalty=float(args.prior_penalty),
        cutoff_objective=str(args.cutoff_objective),
        segment_score_gamma=float(args.segment_score_gamma),
        segment_min_days=int(args.segment_min_days),
        segment_include_open_tail=not bool(args.segment_no_open_tail),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        threshold_step=float(args.threshold_step),
        data_start=str(args.data_start),
        window_days=int(args.window_days),
    )

    plot1 = out_dir / "chart_recall_vs_gap_scatter.png"
    plot2 = out_dir / "chart_metric_distributions.png"
    plot3 = out_dir / "chart_top_recall_profiles.png"
    plot4 = out_dir / "chart_baseline_vs_best.png"
    _plot_recall_vs_gap(df=df, baseline=baseline_row, out_png=plot1)
    _plot_metric_distributions(df=df, baseline=baseline_row, out_png=plot2)
    _plot_top_recall_profiles(top_df=top, out_png=plot3, top_n=min(10, len(top)))
    _plot_baseline_vs_best(best=best, baseline=baseline_row, out_png=plot4)

    delta_recall_min = float(best["test_recall_min"]) - float(baseline_row["test_recall_min"])
    delta_recall_gap = float(best["test_recall_gap"]) - float(baseline_row["test_recall_gap"])
    delta_mcc = float(best["test_mcc"]) - float(baseline_row["test_mcc"])
    delta_acc = float(best["test_acc"]) - float(baseline_row["test_acc"])
    count_ge_baseline = int((df["test_recall_min"] >= float(baseline_row["test_recall_min"])).sum())

    lb_cols = [c for c in _leaderboard_cols() if c in top.columns]
    top_table = top[lb_cols].to_string(index=False)

    trend_df = top.copy()
    trend_df["hidden_sum"] = trend_df["hidden_dims"].astype(str).map(lambda v: float(sum(_parse_hidden_dims(v))))
    corr_hidden = float(trend_df["hidden_sum"].corr(trend_df["test_recall_min"]))
    corr_dropout = float(pd.to_numeric(trend_df["dropout"], errors="coerce").corr(pd.to_numeric(trend_df["test_recall_min"], errors="coerce")))
    top_rec60 = trend_df[trend_df["test_recall_min"] >= 0.60].copy()
    top_rec60_shapes = ", ".join(sorted(set(top_rec60["hidden_dims"].astype(str).tolist()))) if len(top_rec60) else "none"
    top_rec60_drop = ", ".join(sorted({f"{float(v):.2f}" for v in top_rec60["dropout"].tolist()})) if len(top_rec60) else "none"
    lag_value = float(price_ev["event_mean_abs_lag_days"])
    lag_text = "n/a" if not np.isfinite(lag_value) else f"{lag_value:.2f}"
    uses_cutoff_score = "test_cutoff_score" in df.columns
    ranking_line = (
        "- Ranking objective: maximize `test_cutoff_score` "
        "(segment/recall objective with class-gap penalties), then maximize "
        "`test_recall_min`, then minimize `test_recall_gap`, then maximize `test_mcc`, then `test_acc`."
        if uses_cutoff_score
        else "- Ranking objective: maximize `test_recall_min`, then minimize `test_recall_gap`, then maximize `test_mcc`, then `test_acc`."
    )
    practical_takeaway = (
        "- Practical takeaway: trend is now measured under segment-weighted TP reward; compare both `test_cutoff_score` and `test_recall_gap` jointly."
        if uses_cutoff_score
        else "- Practical takeaway: higher dropout trend is visible, but shrink-to-small network trend is not confirmed for `recall_min` objective."
    )
    next_objective_line = (
        "  - preserve objective: maximize `test_cutoff_score` (segment-weighted), keep `test_recall_gap` under control"
        if uses_cutoff_score
        else "  - preserve objective: maximize `recall_min`, then minimize `recall_gap`"
    )
    baseline_cutoff = float(baseline_row.get("test_cutoff_score", np.nan))
    best_cutoff = float(best.get("test_cutoff_score", np.nan))
    cutoff_delta = best_cutoff - baseline_cutoff if (np.isfinite(best_cutoff) and np.isfinite(baseline_cutoff)) else np.nan

    report_lines = [
        "# Notebook-Style Post-Run Report (Winner DCN Grid Search)",
        "",
        f"- Source CSV: `{grid_csv}`",
        f"- Top15 CSV: `{top15_csv}`",
        f"- Meta JSON: `{meta_json}`",
        f"- Replay dataset path: `{dataset_path}`",
        f"- Baseline selected CSV: `{baseline_csv}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        ranking_line,
        "",
        "## Baseline vs New Best",
        "",
        f"- Baseline model: `{baseline_row['model']}` (`{baseline_row['model_type']}`), seed `{int(baseline_row['seed'])}`",
        f"- Baseline test: recall_min=`{_f4(baseline_row['test_recall_min'])}`, recall_gap=`{_f4(baseline_row['test_recall_gap'])}`, MCC=`{_f4(baseline_row['test_mcc'])}`, ACC=`{_f4(baseline_row['test_acc'])}`",
        "",
        f"- New best run_id `{int(best['run_id'])}`, seed `{int(best['seed'])}`, dims `{best['hidden_dims']}`",
        f"- New best test: recall_min=`{_f4(best['test_recall_min'])}`, recall_gap=`{_f4(best['test_recall_gap'])}`, MCC=`{_f4(best['test_mcc'])}`, ACC=`{_f4(best['test_acc'])}`",
        "",
        f"- Delta recall_min: `{delta_recall_min:+.4f}`",
        f"- Delta recall_gap: `{delta_recall_gap:+.4f}` (negative is better)",
        f"- Delta MCC: `{delta_mcc:+.4f}`",
        f"- Delta ACC: `{delta_acc:+.4f}`",
        (
            f"- Delta cutoff_score: `{cutoff_delta:+.4f}`"
            if np.isfinite(cutoff_delta)
            else "- Delta cutoff_score: `n/a` (baseline table has no cutoff_score)"
        ),
        "",
        "## Grid Coverage Summary",
        "",
        f"- Total runs: `{len(df)}`",
        f"- Runs with `test_recall_min >= 0.5`: `{int((df['test_recall_min'] >= 0.5).sum())}`",
        f"- Runs with `test_recall_min >= 0.6`: `{int((df['test_recall_min'] >= 0.6).sum())}`",
        f"- Runs with `test_recall_min >= baseline ({_f4(baseline_row['test_recall_min'])})`: `{count_ge_baseline}`",
        "",
        "## Charts",
        "",
        "### 1) Recall-Min vs Recall-Gap (all runs)",
        "",
        f"![recall-vs-gap]({plot1.name})",
        "",
        "### 2) Metric Distributions",
        "",
        f"![metric-distributions]({plot2.name})",
        "",
        "### 3) Top Recall Profiles",
        "",
        f"![top-recall-profiles]({plot3.name})",
        "",
        "### 4) Baseline vs New Best",
        "",
        f"![baseline-vs-best]({plot4.name})",
        "",
        "### 5) Test Price Markup (Best Run Replay)",
        "",
        f"- Event recall (true): `{float(price_ev['event_recall_true']):.4f}`",
        f"- Event precision (pred): `{float(price_ev['event_precision_pred']):.4f}`",
        f"- Mean abs lag (days): `{lag_text}`",
        f"- Events: true=`{int(price_ev['n_true_events'])}`, pred=`{int(price_ev['n_pred_events'])}`, matched=`{int(price_ev['n_matched_events'])}`",
        "",
        f"![price-event-alignment-best]({Path(str(price_ev['plot_path'])).name})",
        "",
        "## Conclusions & Next Grid",
        "",
        f"- Top-{len(top)} trend corr(`hidden_sum`, `recall_min`) = `{corr_hidden:+.3f}`.",
        f"- Top-{len(top)} trend corr(`dropout`, `recall_min`) = `{corr_dropout:+.3f}`.",
        f"- Runs with `recall_min >= 0.60` in current table: hidden_dims `{top_rec60_shapes}`, dropout `{top_rec60_drop}`.",
        practical_takeaway,
        "- Next directional grid to test your hypothesis safely:",
        "  - hidden_dims: `192x96`, `256x128`, `320x160`, `384x192x96`",
        "  - dropout: `0.40, 0.45, 0.50, 0.55, 0.60`",
        "  - keep DCN and test both `cross 2..6` with `rank 32..96`",
        next_objective_line,
        "",
        "## Leaderboard (Top 15)",
        "",
        f"- Saved table: `{copied_top15 if copied_top15 is not None else top15_csv}`",
        "",
        "```text",
        top_table,
        "```",
    ]

    (out_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[winner_grid_report] out_dir={out_dir}")
    print(f"[winner_grid_report] report={out_dir / 'REPORT.md'}")
    if copied_grid is not None:
        print(f"[winner_grid_report] copied_grid={copied_grid}")
    if copied_top15 is not None:
        print(f"[winner_grid_report] copied_top15={copied_top15}")
    if copied_meta is not None:
        print(f"[winner_grid_report] copied_meta={copied_meta}")


if __name__ == "__main__":
    main()
