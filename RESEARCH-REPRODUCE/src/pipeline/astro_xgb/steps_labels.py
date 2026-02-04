"""
Balanced labels step.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
from typing import Optional

import pandas as pd

import numpy as np

from src.labeling.balanced import build_balanced_labels, gaussian_smooth_centered
from src.labeling.oracle import create_oracle_labels, estimate_threshold_for_move_balance, analyze_label_distribution
from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext
from src.pipeline.astro_xgb.naming import labels_tag
from src.visualization.dxcharts_report import write_dxcharts_report


def _label_spec(label_mode: str) -> tuple[list[int], list[str], dict[int, str]]:
    """
    Determine expected classes based on label_mode.
    """
    if str(label_mode).lower().startswith("oracle"):
        return [0, 1, 2], ["DOWN", "SIDEWAYS", "UP"], {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}
    return [0, 1], ["DOWN", "UP"], {0: "DOWN", 1: "UP"}


def _plot_label_distribution(df_labels: pd.DataFrame, out_path: Path, label_mode: str) -> None:
    """
    Plot class distribution as a bar chart.
    """
    import matplotlib.pyplot as plt

    expected, labels, _ = _label_spec(label_mode)
    counts = df_labels["target"].value_counts(normalize=True).sort_index() * 100
    counts = counts.reindex(expected, fill_value=0.0)
    plt.figure(figsize=(6, 4))
    colors = ["#d62728", "#7f7f7f", "#2ca02c"][: len(labels)]
    plt.bar(labels, counts.values, color=colors)
    plt.title("Class distribution (%)")
    plt.ylabel("%")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _open_file(path: Path) -> None:
    """
    Open a file in the default system viewer.
    """
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"[LABELS] WARNING: Failed to open {path}: {exc}")


def _plot_price_with_labels(
    df_labels: pd.DataFrame,
    price_mode: str,
    out_path: Path,
    title: str = "Price with label shading",
    label_mode: str = "balanced_detrended",
) -> None:
    """
    Plot price with background shading for labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plot_df = df_labels[["date", "close", "target"]].copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df.sort_values("date").reset_index(drop=True)

    if price_mode == "log":
        close = np.log(plot_df["close"].to_numpy())
        y_label = "log(price)"
    else:
        close = plot_df["close"].to_numpy()
        y_label = "price"

    dates = plot_df["date"].to_numpy()
    labels = plot_df["target"].to_numpy()
    expected, _, _ = _label_spec(label_mode)
    if expected == [0, 1]:
        down_mask = labels == 0
        up_mask = labels == 1
        side_mask = None
    else:
        down_mask = labels == 0
        side_mask = labels == 1
        up_mask = labels == 2

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, close, color="black", linewidth=1.0, label="Price")
    if "smoothed_close" in plot_df.columns:
        smooth = plot_df["smoothed_close"].to_numpy()
        if price_mode == "log":
            smooth = np.log(smooth)
        ax.plot(dates, smooth, color="#1f77b4", linewidth=1.0, label="Gaussian")
    ax.fill_between(
        dates, 0, 1, where=down_mask, transform=ax.get_xaxis_transform(),
        color="#d62728", alpha=0.12, label="DOWN"
    )
    if side_mask is not None:
        ax.fill_between(
            dates, 0, 1, where=side_mask, transform=ax.get_xaxis_transform(),
            color="#7f7f7f", alpha=0.12, label="SIDEWAYS"
        )
    ax.fill_between(
        dates, 0, 1, where=up_mask, transform=ax.get_xaxis_transform(),
        color="#2ca02c", alpha=0.12, label="UP"
    )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _grid_search_oracle_quantile(
    df_market: pd.DataFrame,
    sigma_min: int,
    sigma_max: int,
    steps: int,
    q_min: float,
    q_max: float,
    price_mode: str,
) -> pd.DataFrame:
    """
    Grid search where thresholds are picked from quantiles of abs(slope).
    This ensures SIDEWAYS is non-trivial across the grid.
    """
    import numpy as np

    sigmas = np.unique(np.linspace(sigma_min, sigma_max, steps, dtype=int))
    q_grid = np.linspace(q_min, q_max, steps)

    rows = []
    for sigma in sigmas:
        tmp = create_oracle_labels(
            df_market,
            sigma=int(sigma),
            threshold=0.0,
            price_col="close",
            price_mode=price_mode,
            binary_trend=False,
        )
        slope = tmp["smooth_slope"].to_numpy()
        abs_slope = np.abs(slope[np.isfinite(slope)])
        if abs_slope.size == 0:
            continue
        thresholds = np.quantile(abs_slope, q_grid)
        for thr in thresholds:
            labeled = create_oracle_labels(
                df_market,
                sigma=int(sigma),
                threshold=float(thr),
                price_col="close",
                price_mode=price_mode,
                binary_trend=False,
            )
            counts = labeled["target"].value_counts(normalize=True)
            rows.append({
                "sigma": int(sigma),
                "threshold": float(thr),
                "down_pct": counts.get(0, 0) * 100,
                "sideways_pct": counts.get(1, 0) * 100,
                "up_pct": counts.get(2, 0) * 100,
                "imbalance": abs(counts.get(0, 0) - counts.get(2, 0)) * 100,
            })

    return pd.DataFrame(rows).sort_values("imbalance")


def _grid_search_balanced(
    df_market: pd.DataFrame,
    label_mode: str,
    price_mode: str,
    horizon: int,
    gauss_windows: list[int],
    gauss_stds: list[float],
    move_shares: list[float],
    edge_fill: bool,
) -> pd.DataFrame:
    """
    Grid search for balanced labels (binary).
    """
    rows = []
    total = len(df_market)
    for gw in gauss_windows:
        for gs in gauss_stds:
            for ms in move_shares:
                df_labels = build_balanced_labels(
                    df_market=df_market,
                    horizon=horizon,
                    target_move_share=ms,
                    label_mode=label_mode,
                    price_mode=price_mode,
                    gauss_window=gw,
                    gauss_std=gs,
                    price_col="close",
                    edge_fill=edge_fill,
                )
                counts = df_labels["target"].value_counts(normalize=True)
                down_pct = counts.get(0, 0) * 100
                up_pct = counts.get(1, 0) * 100
                rows.append({
                    "gauss_window": int(gw),
                    "gauss_std": float(gs),
                    "move_share": float(ms),
                    "down_pct": down_pct,
                    "up_pct": up_pct,
                    "imbalance": abs(down_pct - up_pct),
                    "rows": len(df_labels),
                    "share_of_total": (len(df_labels) / total * 100) if total else 0.0,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["imbalance", "share_of_total", "rows"], ascending=[True, False, False]).reset_index(drop=True)


def run_labels_step(
    ctx: PipelineContext,
    market_daily_path: Path,
    force: bool = False,
    label_mode: Optional[str] = None,
    price_mode: Optional[str] = None,
    horizon: Optional[int] = None,
    gauss_window: Optional[int] = None,
    gauss_std: Optional[float] = None,
    target_move_share: Optional[float] = None,
    sigma: Optional[float] = None,
    threshold: Optional[float] = None,
    threshold_mode: Optional[str] = None,
    grid_search: bool = False,
    grid_threshold_mode: Optional[str] = None,
    grid_sigma_min: Optional[float] = None,
    grid_sigma_max: Optional[float] = None,
    grid_threshold_min: Optional[float] = None,
    grid_threshold_max: Optional[float] = None,
    grid_steps: Optional[int] = None,
    grid_quantile_min: Optional[float] = None,
    grid_quantile_max: Optional[float] = None,
    balanced_grid_search: bool = False,
    balanced_grid_apply_best: bool = False,
    balanced_grid_windows: Optional[str] = None,
    balanced_grid_stds: Optional[str] = None,
    balanced_grid_move_shares: Optional[str] = None,
    write_report: Optional[bool] = None,
    open_plots: Optional[bool] = None,
    balanced_edge_fill: Optional[bool] = None,
    report_sample_every_days: Optional[int] = None,
    report_max_points: Optional[int] = None,
) -> Path:
    """
    Create balanced labels and return parquet path.
    """
    cfg = ctx.cfg_labels.get("labels", {})

    label_mode = str(label_mode or cfg.get("label_mode", "oracle_gauss"))
    price_mode = str(price_mode or cfg.get("price_mode", "raw"))
    horizon = int(horizon or cfg.get("horizon", 1))
    gauss_window = int(gauss_window or cfg.get("gauss_window", 201))
    gauss_std = float(gauss_std or cfg.get("gauss_std", 50.0))
    target_move_share = float(target_move_share or cfg.get("target_move_share", 0.5))
    sigma = float(sigma or cfg.get("sigma", 3))
    threshold = float(threshold or cfg.get("threshold", 0.0005))
    threshold_mode = str(threshold_mode or cfg.get("threshold_mode", "auto"))
    grid_search = bool(cfg.get("grid_search", False) if grid_search is False else grid_search)
    grid_threshold_mode = str(grid_threshold_mode or cfg.get("grid_threshold_mode", "quantile")).lower()
    grid_sigma_min = float(grid_sigma_min or cfg.get("grid_sigma_min", 2))
    grid_sigma_max = float(grid_sigma_max or cfg.get("grid_sigma_max", 8))
    grid_threshold_min = float(grid_threshold_min or cfg.get("grid_threshold_min", 0.0001))
    grid_threshold_max = float(grid_threshold_max or cfg.get("grid_threshold_max", 0.001))
    grid_steps = int(grid_steps or cfg.get("grid_steps", 7))
    grid_quantile_min = float(grid_quantile_min or cfg.get("grid_quantile_min", 0.1))
    grid_quantile_max = float(grid_quantile_max or cfg.get("grid_quantile_max", 0.9))
    balanced_grid_search = bool(cfg.get("balanced_grid_search", False) if not balanced_grid_search else balanced_grid_search)
    balanced_grid_apply_best = bool(
        cfg.get("balanced_grid_apply_best", False) if not balanced_grid_apply_best else balanced_grid_apply_best
    )
    write_report = bool(cfg.get("write_report", False) if write_report is None else write_report)
    open_plots = bool(cfg.get("open_plots", True) if open_plots is None else open_plots)
    report_sample_every_days = int(
        report_sample_every_days or cfg.get("report_sample_every_days", 1)
    )
    report_max_points = int(report_max_points or cfg.get("report_max_points", 0))
    balanced_edge_fill = bool(cfg.get("balanced_edge_fill", True) if balanced_edge_fill is None else balanced_edge_fill)

    def _parse_list(value: Optional[str], default: list[float]) -> list[float]:
        if not value:
            return default
        return [float(v.strip()) for v in value.split(",") if v.strip()]

    default_windows = [int(v) for v in cfg.get("balanced_grid_windows", [101, 151, 201])]
    default_stds = [float(v) for v in cfg.get("balanced_grid_stds", [30, 50, 70])]
    default_moves = [float(v) for v in cfg.get("balanced_grid_move_shares", [0.2, 0.3, 0.4, 0.5])]

    def _ensure_odd_windows(values: list[int], label: str) -> list[int]:
        cleaned: list[int] = []
        for raw in values:
            val = int(raw)
            if val % 2 == 0:
                adj = val + 1
                print(f"[LABELS] WARNING: {label} window {val} is even; using {adj}.")
                val = adj
            cleaned.append(val)
        # Preserve order, remove duplicates
        return list(dict.fromkeys(cleaned))

    if gauss_window % 2 == 0:
        adj = gauss_window + 1
        print(f"[LABELS] WARNING: gauss_window {gauss_window} is even; using {adj}.")
        gauss_window = adj

    grid_windows = _ensure_odd_windows(
        [int(v) for v in _parse_list(balanced_grid_windows, default_windows)],
        label="grid",
    )
    grid_stds = [float(v) for v in _parse_list(balanced_grid_stds, default_stds)]
    grid_moves = [float(v) for v in _parse_list(balanced_grid_move_shares, default_moves)]

    df_market = pd.read_parquet(market_daily_path)
    df_market["date"] = pd.to_datetime(df_market["date"])

    # Balanced grid search (binary modes)
    if balanced_grid_search:
        if label_mode.lower().startswith("balanced"):
            grid_df = _grid_search_balanced(
                df_market=df_market,
                label_mode=label_mode,
                price_mode=price_mode,
                horizon=horizon,
                gauss_windows=grid_windows,
                gauss_stds=grid_stds,
                move_shares=grid_moves,
                edge_fill=balanced_edge_fill,
            )
            grid_path = ctx.reports_dir / f"{ctx.subject.subject_id}_labels_balanced_grid.csv"
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            grid_df.to_csv(grid_path, index=False)
            print("[LABELS] Balanced grid search saved:", grid_path)
            print("[LABELS] Top 10 balanced combos:")
            print(grid_df.head(10))
            if balanced_grid_apply_best and not grid_df.empty:
                best = grid_df.iloc[0].to_dict()
                gauss_window = int(best["gauss_window"])
                gauss_std = float(best["gauss_std"])
                target_move_share = float(best["move_share"])
                print(
                    "[LABELS] Applying best balanced params:",
                    f"gauss_window={gauss_window}, gauss_std={gauss_std}, target_move_share={target_move_share}",
                )
        else:
            print("[LABELS] Balanced grid search is only for balanced label modes; skipping.")

    if label_mode.lower().startswith("oracle") and threshold_mode.lower() == "auto":
        threshold = estimate_threshold_for_move_balance(
            df_market,
            sigma=int(sigma),
            price_col="close",
            price_mode=price_mode,
            target_move_share=target_move_share,
            min_threshold=cfg.get("threshold_min", 0.0),
            max_threshold=cfg.get("threshold_max", None),
        )

    tag = labels_tag(
        mode=label_mode,
        horizon=horizon,
        price_mode=price_mode,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
        move_share=target_move_share,
        sigma=sigma,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )
    out_path = ctx.processed_dir / f"{ctx.subject.subject_id}_labels_{tag}.parquet"

    params = {
        "label_mode": label_mode,
        "price_mode": price_mode,
        "horizon": horizon,
        "gauss_window": gauss_window,
        "gauss_std": gauss_std,
        "target_move_share": target_move_share,
        "sigma": sigma,
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "balanced_edge_fill": balanced_edge_fill,
    }
    inputs = [Path(market_daily_path)]

    if not force and is_cache_valid(out_path, params=params, inputs=inputs, step="labels"):
        df_labels = pd.read_parquet(out_path)
        print("[LABELS] Using cached labels:", out_path)
    else:
        if label_mode.lower().startswith("oracle"):
            df_labels = create_oracle_labels(
                df_market,
                sigma=int(sigma),
                threshold=float(threshold),
                price_col="close",
                price_mode=price_mode,
                binary_trend=False,
            )
            # Shift target by horizon so features predict future label
            if horizon > 0:
                df_labels = df_labels.copy()
                df_labels["target"] = df_labels["target"].shift(-horizon)
                df_labels = df_labels.dropna(subset=["target"]).reset_index(drop=True)
                df_labels["target"] = df_labels["target"].astype(int)
        else:
            if label_mode.lower() == "balanced_detrended":
                base = df_market["close"].astype(float)
                if price_mode == "log":
                    base = np.log(base)
                smooth = gaussian_smooth_centered(base, gauss_window, gauss_std)
                if price_mode == "log":
                    df_market["smoothed_close"] = np.exp(smooth)
                else:
                    df_market["smoothed_close"] = smooth
            df_labels = build_balanced_labels(
                df_market=df_market,
                horizon=horizon,
                target_move_share=target_move_share,
                label_mode=label_mode,
                price_mode=price_mode,
                gauss_window=gauss_window,
                gauss_std=gauss_std,
                price_col="close",
                edge_fill=balanced_edge_fill,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_labels.to_parquet(out_path, index=False)

        meta = build_meta(step="labels", params=params, inputs=inputs)
        save_meta(meta_path_for(out_path), meta)

        print("[LABELS] Built labels.")
        print("[LABELS] Saved:", out_path)

    if label_mode.lower() == "balanced_detrended" and "smoothed_close" not in df_labels.columns:
        base = df_market["close"].astype(float)
        if price_mode == "log":
            base = np.log(base)
        smooth = gaussian_smooth_centered(base, gauss_window, gauss_std)
        if balanced_edge_fill:
            smooth = smooth.bfill().ffill()
        if price_mode == "log":
            smooth = np.exp(smooth)
        smooth_df = pd.DataFrame({"date": df_market["date"], "smoothed_close": smooth})
        df_labels = df_labels.merge(smooth_df, on="date", how="left")

    print("[LABELS] Params:", params)
    if not df_labels.empty:
        print(f"[LABELS] Rows: {len(df_labels)} | "
              f"Range: {df_labels['date'].min().date()} -> {df_labels['date'].max().date()}")
        expected, _, _ = _label_spec(label_mode)
        dist = df_labels["target"].value_counts(normalize=True).sort_index() * 100
        dist = dist.reindex(expected, fill_value=0.0)
        print("[LABELS] Class share (%):", {int(k): round(v, 2) for k, v in dist.items()})
        missing = [k for k, v in dist.items() if v == 0.0]
        if missing:
            print("[LABELS] WARNING: missing classes:", missing)

    # Plot distribution + price shading
    dist_path = ctx.reports_dir / f"{ctx.subject.subject_id}_labels_dist_{tag}.png"
    shade_path = ctx.reports_dir / f"{ctx.subject.subject_id}_labels_shaded_{tag}.png"
    _plot_label_distribution(df_labels, dist_path, label_mode=label_mode)
    _plot_price_with_labels(df_labels, price_mode=price_mode, out_path=shade_path, label_mode=label_mode)
    print("[LABELS] Distribution plot:", dist_path)
    print("[LABELS] Shaded price plot:", shade_path)

    # Optional grid search for sigma/threshold balance (oracle only)
    if grid_search and label_mode.lower().startswith("oracle"):
        if grid_threshold_mode == "quantile":
            grid_df = _grid_search_oracle_quantile(
                df_market=df_market,
                sigma_min=int(grid_sigma_min),
                sigma_max=int(grid_sigma_max),
                steps=int(grid_steps),
                q_min=float(grid_quantile_min),
                q_max=float(grid_quantile_max),
                price_mode=price_mode,
            )
        else:
            grid_df = analyze_label_distribution(
                df_market,
                sigma_range=(int(grid_sigma_min), int(grid_sigma_max)),
                threshold_range=(float(grid_threshold_min), float(grid_threshold_max)),
                n_steps=int(grid_steps),
                price_mode=price_mode,
                price_col="close",
            )

        if not grid_df.empty and grid_df["sideways_pct"].max() < 1.0:
            print("[LABELS] WARNING: grid has almost no SIDEWAYS. "
                  "Try --grid-threshold-mode quantile or increase threshold range.")
        grid_path = ctx.reports_dir / f"{ctx.subject.subject_id}_labels_grid_{tag}.parquet"
        grid_csv = ctx.reports_dir / f"{ctx.subject.subject_id}_labels_grid_{tag}.csv"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_parquet(grid_path, index=False)
        grid_df.to_csv(grid_csv, index=False)
        print("[LABELS] Grid search saved:", grid_path)
        print("[LABELS] Grid search CSV:", grid_csv)
        print("[LABELS] Top 10 by balance:")
        print(grid_df.head(10))

    if write_report:
        report_path = ctx.reports_dir / f"{ctx.subject.subject_id}_labels_{tag}.html"
        write_dxcharts_report(
            df_labels,
            report_path,
            title=f"Labels: {ctx.subject.subject_id}",
            extra_images=[dist_path, shade_path],
            sample_every_days=report_sample_every_days,
            max_points=report_max_points,
        )
        print("[LABELS] Report:", report_path)
        if open_plots:
            _open_file(report_path)

    return out_path
