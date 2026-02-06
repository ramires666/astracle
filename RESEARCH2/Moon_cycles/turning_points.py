"""
Turning-point labeling helpers for Moon-cycle research.

This module implements a simple, transparent first-step protocol:
1) Mark candidate days where price can hit a target move up/down
   within a fixed future horizon.
2) Compress dense candidates into sparse turning points.
3) Build regime intervals for plotting and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurningPointLabelConfig:
    """
    Parameters for trend-change point labeling.

    horizon_days:
        Look-ahead horizon for checking if target move is reached.
    up_move_pct / down_move_pct:
        Required move amplitude (e.g. 0.09 means 9%).
    cluster_gap_days:
        Nearby same-direction candidates are grouped into one cluster.
    min_turn_gap_days:
        Minimal distance between accepted turning points.
    past_horizon_days:
        Optional look-back horizon for reversal confirmation.
        If > 0, an UP turn also needs a prior DOWN move and vice versa.
    past_up_move_pct / past_down_move_pct:
        Optional past confirmation amplitudes. If None, reuse forward
        up_move_pct/down_move_pct respectively.
    """

    horizon_days: int = 10
    up_move_pct: float = 0.09
    down_move_pct: float = 0.09
    cluster_gap_days: int = 2
    min_turn_gap_days: int = 2
    past_horizon_days: int = 0
    past_up_move_pct: float | None = None
    past_down_move_pct: float | None = None
    # Last open segment handling:
    # - "last_turn": keep direction of last confirmed turn (legacy behavior).
    # - "endpoint_sign": infer by return from last turn price to latest close.
    tail_direction_mode: str = "last_turn"
    # Dead zone for endpoint_sign (e.g. 0.01 means +/-1% keeps fallback).
    tail_min_move_pct: float = 0.0


def _validate_market_df(df_market: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df_market.columns or "close" not in df_market.columns:
        raise ValueError("df_market must contain 'date' and 'close' columns")

    out = df_market[["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


def build_turning_point_candidates(
    df_market: pd.DataFrame,
    cfg: TurningPointLabelConfig,
) -> pd.DataFrame:
    """
    Build day-level candidates based on future threshold hits.

    For each day t:
    - UP candidate if close can reach +up_move_pct within next horizon_days.
    - DOWN candidate if close can reach -down_move_pct within next horizon_days.
    - If both are reachable, earliest hit wins.
    - Optional: require confirmation in the past window:
      * UP requires prior drawdown from a local peak.
      * DOWN requires prior run-up from a local trough.
    """

    if cfg.horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if cfg.up_move_pct <= 0.0 or cfg.down_move_pct <= 0.0:
        raise ValueError("up_move_pct and down_move_pct must be > 0")
    if cfg.past_horizon_days < 0:
        raise ValueError("past_horizon_days must be >= 0")

    use_past_confirmation = int(cfg.past_horizon_days) > 0
    past_up_move_pct = float(
        cfg.past_up_move_pct if cfg.past_up_move_pct is not None else cfg.up_move_pct
    )
    past_down_move_pct = float(
        cfg.past_down_move_pct if cfg.past_down_move_pct is not None else cfg.down_move_pct
    )
    if past_up_move_pct <= 0.0 or past_down_move_pct <= 0.0:
        raise ValueError("past_up_move_pct and past_down_move_pct must be > 0")

    df = _validate_market_df(df_market)
    prices = df["close"].to_numpy(dtype=float)
    n = len(prices)
    horizon = int(cfg.horizon_days)
    past_horizon = int(cfg.past_horizon_days)

    max_future_return = np.full(n, np.nan, dtype=float)
    min_future_return = np.full(n, np.nan, dtype=float)
    up_hit_days = np.full(n, np.nan, dtype=float)
    down_hit_days = np.full(n, np.nan, dtype=float)
    past_max_return = np.full(n, np.nan, dtype=float)
    past_min_return = np.full(n, np.nan, dtype=float)
    past_up_confirm = np.zeros(n, dtype=np.int32)
    past_down_confirm = np.zeros(n, dtype=np.int32)
    direction = np.zeros(n, dtype=np.int32)
    score = np.zeros(n, dtype=float)

    for i in range(n):
        j_end = min(n, i + horizon + 1)
        if i + 1 >= j_end:
            continue

        base = prices[i]
        future = prices[i + 1 : j_end]

        local_max_ret = float(np.max(future) / base - 1.0)
        local_min_ret = float(np.min(future) / base - 1.0)
        max_future_return[i] = local_max_ret
        min_future_return[i] = local_min_ret

        if use_past_confirmation:
            p_start = max(0, i - past_horizon)
            past = prices[p_start:i]
            if past.size > 0:
                # Move to current point from best/worst past anchor.
                local_past_max_ret = float(base / np.min(past) - 1.0)
                local_past_min_ret = float(base / np.max(past) - 1.0)
                past_max_return[i] = local_past_max_ret
                past_min_return[i] = local_past_min_ret
                if local_past_max_ret >= past_up_move_pct:
                    past_up_confirm[i] = 1
                if local_past_min_ret <= -past_down_move_pct:
                    past_down_confirm[i] = 1

        up_target = base * (1.0 + float(cfg.up_move_pct))
        down_target = base * (1.0 - float(cfg.down_move_pct))

        up_d = np.nan
        down_d = np.nan
        for d, px in enumerate(future, start=1):
            if np.isnan(up_d) and px >= up_target:
                up_d = float(d)
            if np.isnan(down_d) and px <= down_target:
                down_d = float(d)
            if not np.isnan(up_d) and not np.isnan(down_d):
                break

        up_hit_days[i] = up_d
        down_hit_days[i] = down_d

        allow_up = not np.isnan(up_d)
        allow_down = not np.isnan(down_d)
        if use_past_confirmation:
            allow_up = allow_up and bool(past_down_confirm[i] == 1)
            allow_down = allow_down and bool(past_up_confirm[i] == 1)

        if not allow_up and not allow_down:
            direction[i] = 0
            score[i] = 0.0
            continue

        if allow_up and not allow_down:
            direction[i] = 1
        elif allow_down and not allow_up:
            direction[i] = -1
        elif up_d < down_d:
            direction[i] = 1
        elif down_d < up_d:
            direction[i] = -1
        else:
            direction[i] = 1 if local_max_ret >= abs(local_min_ret) else -1

        score[i] = local_max_ret if direction[i] == 1 else abs(local_min_ret)

    out = df.copy()
    out["max_future_return"] = max_future_return
    out["min_future_return"] = min_future_return
    out["up_hit_days"] = up_hit_days
    out["down_hit_days"] = down_hit_days
    out["past_max_return"] = past_max_return
    out["past_min_return"] = past_min_return
    out["past_up_confirm"] = past_up_confirm
    out["past_down_confirm"] = past_down_confirm
    out["candidate_direction"] = direction
    out["candidate_score"] = score
    out["turning_point"] = 0
    return out


def _compress_candidates_to_turning_points(
    df_candidates: pd.DataFrame,
    cfg: TurningPointLabelConfig,
) -> pd.DataFrame:
    """
    Compress dense day-level candidates into sparse turning points.

    Strategy:
    - Group nearby same-direction candidates into clusters.
    - Keep strongest score inside each cluster.
    - Enforce alternation and minimal temporal gap.
    """

    cols = [
        "date",
        "close",
        "candidate_direction",
        "candidate_score",
        "max_future_return",
        "min_future_return",
        "up_hit_days",
        "down_hit_days",
        "past_max_return",
        "past_min_return",
        "past_up_confirm",
        "past_down_confirm",
    ]
    non_zero = (
        df_candidates.loc[df_candidates["candidate_direction"] != 0, cols]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if non_zero.empty:
        non_zero["turning_direction"] = []
        return non_zero

    run_ids = np.zeros(len(non_zero), dtype=np.int32)
    run_id = 0
    prev_date = pd.to_datetime(non_zero.loc[0, "date"])
    prev_dir = int(non_zero.loc[0, "candidate_direction"])

    for idx in range(1, len(non_zero)):
        cur_date = pd.to_datetime(non_zero.loc[idx, "date"])
        cur_dir = int(non_zero.loc[idx, "candidate_direction"])
        gap_days = int((cur_date - prev_date).days)
        same_cluster = (cur_dir == prev_dir) and (gap_days <= int(cfg.cluster_gap_days))
        if not same_cluster:
            run_id += 1
        run_ids[idx] = run_id
        prev_date = cur_date
        prev_dir = cur_dir

    non_zero["run_id"] = run_ids
    best_idx = non_zero.groupby("run_id")["candidate_score"].idxmax()
    run_best = non_zero.loc[best_idx].sort_values("date").reset_index(drop=True)

    kept_rows = []
    for row in run_best.itertuples(index=False):
        cur_date = pd.to_datetime(row.date)
        cur_dir = int(row.candidate_direction)
        cur_score = float(row.candidate_score)

        if not kept_rows:
            kept_rows.append(row)
            continue

        prev = kept_rows[-1]
        prev_date = pd.to_datetime(prev.date)
        prev_dir = int(prev.candidate_direction)
        prev_score = float(prev.candidate_score)
        gap_days = int((cur_date - prev_date).days)

        if cur_dir == prev_dir:
            if cur_score > prev_score:
                kept_rows[-1] = row
            continue

        if gap_days < int(cfg.min_turn_gap_days):
            if cur_score > prev_score:
                kept_rows[-1] = row
            continue

        kept_rows.append(row)

    out = pd.DataFrame(kept_rows).copy()
    out = out.sort_values("date").reset_index(drop=True)
    out = out.rename(columns={"candidate_direction": "turning_direction"})
    return out


def build_regime_intervals(
    df_market: pd.DataFrame,
    df_turning_points: pd.DataFrame,
    tail_direction_mode: str = "last_turn",
    tail_min_move_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Build regime intervals from each turning point to the next one.
    """

    base = _validate_market_df(df_market)
    if df_turning_points.empty:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "direction",
                "regime_label",
                "duration_days",
            ]
        )

    tp = df_turning_points.sort_values("date").reset_index(drop=True)
    market_end = pd.to_datetime(base["date"].max())

    market_end_close = float(pd.to_numeric(base["close"], errors="coerce").iloc[-1])

    rows = []
    for i, row in tp.iterrows():
        start_date = pd.to_datetime(row["date"])
        if i + 1 < len(tp):
            end_date = pd.to_datetime(tp.loc[i + 1, "date"])
        else:
            end_date = market_end
        direction = int(row["turning_direction"])
        if i + 1 == len(tp) and str(tail_direction_mode).lower() == "endpoint_sign":
            start_close = pd.to_numeric(pd.Series([row.get("close", np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(float(start_close)) and float(start_close) > 0.0 and market_end_close > 0.0:
                tail_ret = float(market_end_close / float(start_close) - 1.0)
                if abs(tail_ret) >= float(max(0.0, tail_min_move_pct)):
                    direction = 1 if tail_ret > 0.0 else -1
        label = "UP_REGIME" if direction == 1 else "DOWN_REGIME"
        duration_days = max(0, int((end_date - start_date).days))
        rows.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "direction": direction,
                "regime_label": label,
                "duration_days": duration_days,
            }
        )

    return pd.DataFrame(rows)


def label_turning_points(
    df_market: pd.DataFrame,
    cfg: TurningPointLabelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full turning-point labeling pipeline.

    Returns:
    - df_candidates: day-level diagnostics + candidate labels
    - df_turning_points: sparse turning-point table
    - df_regimes: regime intervals for plotting
    """

    df_candidates = build_turning_point_candidates(df_market=df_market, cfg=cfg)
    df_turning_points = _compress_candidates_to_turning_points(df_candidates, cfg=cfg)

    turning_map = dict(
        zip(
            pd.to_datetime(df_turning_points["date"]),
            df_turning_points["turning_direction"].astype(int),
        )
    )
    df_candidates["turning_point"] = (
        pd.to_datetime(df_candidates["date"]).map(turning_map).fillna(0).astype(int)
    )

    df_regimes = build_regime_intervals(
        df_market=df_market,
        df_turning_points=df_turning_points,
        tail_direction_mode=str(getattr(cfg, "tail_direction_mode", "last_turn")),
        tail_min_move_pct=float(getattr(cfg, "tail_min_move_pct", 0.0)),
    )
    return df_candidates, df_turning_points, df_regimes


def plot_turning_points_chart(
    df_market: pd.DataFrame,
    df_turning_points: pd.DataFrame,
    df_regimes: pd.DataFrame | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (16, 7),
    y_scale: str = "linear",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot price, turning points, and shaded regimes.

    Visual style requested:
    - Up turning points: dark green bold marker.
    - Down turning points: dark red bold marker.
    - Up regimes: light green vertical shading.
    - Down regimes: light red vertical shading.
    """

    df = _validate_market_df(df_market)
    scale = str(y_scale).lower()
    if scale not in {"linear", "log"}:
        raise ValueError("y_scale must be 'linear' or 'log'")
    if scale == "log" and bool((df["close"] <= 0).any()):
        raise ValueError("log y-scale requires all close values > 0")

    tp = df_turning_points.copy()
    if not tp.empty:
        tp["date"] = pd.to_datetime(tp["date"])
        tp = tp.sort_values("date").reset_index(drop=True)

    regimes = df_regimes.copy() if df_regimes is not None else pd.DataFrame()
    if not regimes.empty:
        regimes["start_date"] = pd.to_datetime(regimes["start_date"])
        regimes["end_date"] = pd.to_datetime(regimes["end_date"])
        regimes = regimes.sort_values("start_date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Background regime shading.
    if not regimes.empty:
        first_up = True
        first_down = True
        for row in regimes.itertuples(index=False):
            if int(row.direction) == 1:
                ax.axvspan(
                    row.start_date,
                    row.end_date,
                    color="#33BE64C7",
                    alpha=0.28,
                    lw=0,
                    label="UP regime" if first_up else None,
                )
                first_up = False
            else:
                ax.axvspan(
                    row.start_date,
                    row.end_date,
                    color="#CA3030",
                    alpha=0.28,
                    lw=0,
                    label="DOWN regime" if first_down else None,
                )
                first_down = False

    # Price series.
    ax.plot(
        df["date"],
        df["close"],
        color="#1F2937",
        lw=1.6,
        alpha=0.95,
        label="BTC close",
    )

    # Turning points.
    if not tp.empty:
        tp_up = tp[tp["turning_direction"] == 1]
        tp_down = tp[tp["turning_direction"] == -1]

        if not tp_up.empty:
            ax.scatter(
                tp_up["date"],
                tp_up["close"],
                marker="^",
                s=170,
                color="#006400",
                edgecolors="#022C22",
                linewidths=1.2,
                zorder=5,
                label="Turn UP",
            )
        if not tp_down.empty:
            ax.scatter(
                tp_down["date"],
                tp_down["close"],
                marker="v",
                s=170,
                color="#8B0000",
                edgecolors="#3D0000",
                linewidths=1.2,
                zorder=5,
                label="Turn DOWN",
            )

    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.set_yscale(scale)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(title or "Trend turning points and regimes")
    fig.tight_layout()
    return fig, ax
