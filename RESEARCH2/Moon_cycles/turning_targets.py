"""
Target builders for sparse turning-point classification.

All builders return a DataFrame with at least:
- date
- target (0=DOWN turn, 1=UP turn)
- turning_direction (-1 or +1)
- sample_weight
- target_mode

Design goal:
- Keep binary target (UP/DOWN) as requested.
- Provide denser supervision than "event dates only".
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .turning_targets_numba import (
    NUMBA_AVAILABLE,
    segment_midpoint_scan,
    window_kernel_scan,
)

TargetMode = Literal["point_only", "window_kernel", "segment_midpoint"]


def _normalize_market_dates(df_market: pd.DataFrame) -> pd.DataFrame:
    """Return sorted unique date frame from market data."""
    if "date" not in df_market.columns:
        raise ValueError("df_market must contain 'date'")
    out = df_market[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


def _normalize_turning_points(df_turning_points: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize turning-point table."""
    need = {"date", "turning_direction"}
    missing = need - set(df_turning_points.columns)
    if missing:
        raise ValueError(f"df_turning_points missing columns: {sorted(missing)}")

    cols = ["date", "turning_direction"]
    if "close" in df_turning_points.columns:
        cols.append("close")

    out = df_turning_points[cols].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["turning_direction"] = pd.to_numeric(out["turning_direction"], errors="coerce")
    out = out.dropna(subset=["date", "turning_direction"])
    out["turning_direction"] = out["turning_direction"].astype(int)
    out = out[out["turning_direction"].isin([-1, 1])]

    if "close" in out.columns:
        out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


def _direction_to_target(direction: pd.Series | np.ndarray) -> np.ndarray:
    """Map turning direction {-1, +1} to binary target {0, 1}."""
    arr = np.asarray(direction, dtype=np.int32)
    return (arr > 0).astype(np.int32)


def _event_amplitude_scale(df_tp: pd.DataFrame, default_value: float = 1.0) -> np.ndarray:
    """
    Compute per-event amplitude multipliers from price jumps between turns.

    If close is unavailable, return constant weights.
    """
    if "close" not in df_tp.columns or df_tp["close"].isna().all():
        return np.full(len(df_tp), float(default_value), dtype=float)

    close = pd.to_numeric(df_tp["close"], errors="coerce").astype(float)
    move = close.pct_change().abs()

    med = float(np.nanmedian(move.to_numpy(dtype=float)))
    if not np.isfinite(med) or med <= 0.0:
        med = 1.0

    out = (move / med).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=float)
    out = np.clip(out, 0.50, 3.00)
    return out


def build_point_only_targets(
    df_turning_points: pd.DataFrame,
    use_amplitude_weight: bool = True,
) -> pd.DataFrame:
    """Build sparse target: only exact turning dates are supervised."""
    tp = _normalize_turning_points(df_turning_points)
    if tp.empty:
        return pd.DataFrame(columns=["date", "target", "turning_direction", "sample_weight", "target_mode"])

    w = _event_amplitude_scale(tp) if use_amplitude_weight else np.ones(len(tp), dtype=float)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(tp["date"]),
            "turning_direction": tp["turning_direction"].astype(np.int32),
            "target": _direction_to_target(tp["turning_direction"]),
            "sample_weight": w.astype(float),
            "target_mode": "point_only",
        }
    )
    return out.sort_values("date").reset_index(drop=True)


def build_window_kernel_targets(
    df_market: pd.DataFrame,
    df_turning_points: pd.DataFrame,
    radius_days: int = 8,
    distance_power: float = 1.5,
    min_weight: float = 0.05,
    use_amplitude_weight: bool = True,
    use_numba: bool = True,
) -> pd.DataFrame:
    """
    Build denser event-centered targets with distance-decay kernel.

    Each turning point contributes labels in +-radius_days window.
    In overlaps we keep the higher-weight event.
    """
    if radius_days < 0:
        raise ValueError("radius_days must be >= 0")
    if distance_power <= 0.0:
        raise ValueError("distance_power must be > 0")

    base = _normalize_market_dates(df_market)
    tp = _normalize_turning_points(df_turning_points)

    if base.empty or tp.empty:
        return pd.DataFrame(columns=["date", "target", "turning_direction", "sample_weight", "target_mode"])

    idx_by_date = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(base["date"]))}
    n = len(base)

    tp_work = tp.reset_index(drop=True).copy()
    tp_work["center_idx"] = [idx_by_date.get(pd.Timestamp(d), -1) for d in pd.to_datetime(tp_work["date"])]
    tp_work = tp_work[tp_work["center_idx"] >= 0].reset_index(drop=True)
    if tp_work.empty:
        return pd.DataFrame(columns=["date", "target", "turning_direction", "sample_weight", "target_mode"])

    amp = _event_amplitude_scale(tp_work) if use_amplitude_weight else np.ones(len(tp_work), dtype=float)

    best_weight, best_dir, best_event = window_kernel_scan(
        center_idx=tp_work["center_idx"].to_numpy(dtype=np.int64),
        direction=tp_work["turning_direction"].to_numpy(dtype=np.int32),
        event_amp=np.asarray(amp, dtype=np.float64),
        n_rows=int(n),
        radius_days=int(radius_days),
        distance_power=float(distance_power),
        use_numba=bool(use_numba),
    )

    mask = best_event >= 0
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(base.loc[mask, "date"]).reset_index(drop=True),
            "turning_direction": best_dir[mask].astype(np.int32),
            "sample_weight": np.clip(best_weight[mask], float(min_weight), None).astype(float),
            "event_index": best_event[mask].astype(np.int32),
            "target_mode": "window_kernel",
        }
    )
    out["target"] = _direction_to_target(out["turning_direction"])
    return out.sort_values("date").reset_index(drop=True)


def build_segment_midpoint_targets(
    df_market: pd.DataFrame,
    df_turning_points: pd.DataFrame,
    center_power: float = 1.5,
    min_weight: float = 0.05,
    use_amplitude_weight: bool = True,
    use_numba: bool = True,
    segment_direction_anchor: Literal["current_turn", "next_turn"] = "current_turn",
    include_last_open_segment: bool = True,
    open_tail_direction_mode: Literal["last_turn", "endpoint_sign"] = "last_turn",
    open_tail_min_move_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Build dense segment target with maximum weight at segment center.

    Segment definition:
    - Segments are non-overlapping by construction.
    - `segment_direction_anchor="current_turn"`:
      segment i is [turn_i, turn_{i+1}) in time.
    - `segment_direction_anchor="next_turn"`:
      segment i is (turn_i, turn_{i+1}] in time.
    - `segment_direction_anchor="current_turn"`:
      class is direction of turn_i (same semantics as base regime shading).
    - `segment_direction_anchor="next_turn"`:
      class is direction of turn_{i+1} (legacy predictive semantics).
    - If `include_last_open_segment=True` and anchor is `current_turn`,
      open tail direction can be:
      * `open_tail_direction_mode="last_turn"`: direction of last turn.
      * `open_tail_direction_mode="endpoint_sign"`: sign(last_close / last_turn_close - 1),
        with dead-zone fallback to last-turn direction.

    This follows the user idea: keep point-to-point structure and emphasize
    center area of each range via sample_weight peak.
    """
    if center_power <= 0.0:
        raise ValueError("center_power must be > 0")
    if segment_direction_anchor not in {"current_turn", "next_turn"}:
        raise ValueError("segment_direction_anchor must be 'current_turn' or 'next_turn'")
    if open_tail_direction_mode not in {"last_turn", "endpoint_sign"}:
        raise ValueError("open_tail_direction_mode must be 'last_turn' or 'endpoint_sign'")

    base = _normalize_market_dates(df_market)
    tp = _normalize_turning_points(df_turning_points)

    if len(tp) < 2 or base.empty:
        return pd.DataFrame(columns=["date", "target", "turning_direction", "sample_weight", "target_mode"])

    idx_by_date = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(base["date"]))}
    n = len(base)

    # Segment amplitude scales from move between neighboring turning points.
    if "close" in tp.columns and use_amplitude_weight:
        close = pd.to_numeric(tp["close"], errors="coerce")
        seg_move = ((close.shift(-1) / close - 1.0).abs().fillna(0.0).to_numpy(dtype=float))
        med = float(np.nanmedian(seg_move[:-1])) if len(seg_move) > 1 else 0.0
        if not np.isfinite(med) or med <= 0.0:
            med = 1.0
        seg_amp = np.clip(seg_move / med, 0.50, 3.00)
    else:
        seg_amp = np.ones(len(tp), dtype=float)

    start_idx_arr: list[int] = []
    end_idx_arr: list[int] = []
    seg_dir_arr: list[int] = []
    seg_amp_arr: list[float] = []

    for i in range(len(tp) - 1):
        start_idx = idx_by_date.get(pd.Timestamp(tp.loc[i, "date"]), -1)
        end_idx = idx_by_date.get(pd.Timestamp(tp.loc[i + 1, "date"]), -1)
        if start_idx < 0 or end_idx < 0:
            continue

        # Avoid boundary overlap between neighboring segments:
        # - current_turn: [turn_i, turn_{i+1})
        # - next_turn:    (turn_i, turn_{i+1}]
        if segment_direction_anchor == "next_turn":
            seg_start = int(start_idx) + 1
            seg_end = int(end_idx)
        else:
            seg_start = int(start_idx)
            seg_end = int(end_idx) - 1

        if seg_end < seg_start:
            continue

        start_idx_arr.append(seg_start)
        end_idx_arr.append(seg_end)
        if segment_direction_anchor == "next_turn":
            seg_dir_arr.append(int(tp.loc[i + 1, "turning_direction"]))
        else:
            seg_dir_arr.append(int(tp.loc[i, "turning_direction"]))
        seg_amp_arr.append(float(seg_amp[i]))

    # Optional tail labeling for current-regime semantics:
    # keep last known regime active until latest known market date.
    if include_last_open_segment and segment_direction_anchor == "current_turn" and len(tp) >= 1:
        last_start_idx = idx_by_date.get(pd.Timestamp(tp.loc[len(tp) - 1, "date"]), -1)
        last_end_idx = int(n - 1)
        if 0 <= last_start_idx <= last_end_idx:
            tail_dir = int(tp.loc[len(tp) - 1, "turning_direction"])
            if open_tail_direction_mode == "endpoint_sign":
                start_close = np.nan
                if "close" in tp.columns:
                    start_close = float(
                        pd.to_numeric(pd.Series([tp.loc[len(tp) - 1, "close"]]), errors="coerce").iloc[0]
                    )
                end_close = np.nan
                if "close" in df_market.columns:
                    end_close_series = pd.to_numeric(df_market["close"], errors="coerce").dropna()
                    if len(end_close_series) > 0:
                        end_close = float(end_close_series.iloc[-1])

                if np.isfinite(start_close) and start_close > 0.0 and np.isfinite(end_close) and end_close > 0.0:
                    tail_ret = float(end_close / start_close - 1.0)
                    if abs(tail_ret) >= float(max(0.0, open_tail_min_move_pct)):
                        tail_dir = 1 if tail_ret > 0.0 else -1

            start_idx_arr.append(int(last_start_idx))
            end_idx_arr.append(int(last_end_idx))
            seg_dir_arr.append(int(tail_dir))

            if len(seg_amp_arr) > 0:
                tail_amp = float(seg_amp_arr[-1])
            else:
                tail_amp = 1.0
            seg_amp_arr.append(float(tail_amp))

    if len(start_idx_arr) == 0:
        return pd.DataFrame(columns=["date", "target", "turning_direction", "sample_weight", "target_mode"])

    best_weight, best_dir, best_segment = segment_midpoint_scan(
        start_idx=np.asarray(start_idx_arr, dtype=np.int64),
        end_idx=np.asarray(end_idx_arr, dtype=np.int64),
        direction=np.asarray(seg_dir_arr, dtype=np.int32),
        seg_amp=np.asarray(seg_amp_arr, dtype=np.float64),
        n_rows=int(n),
        center_power=float(center_power),
        min_weight=float(min_weight),
        use_numba=bool(use_numba),
    )

    mask = best_segment >= 0
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(base.loc[mask, "date"]).reset_index(drop=True),
            "turning_direction": best_dir[mask].astype(np.int32),
            "sample_weight": best_weight[mask].astype(float),
            "segment_index": best_segment[mask].astype(np.int32),
            "target_mode": "segment_midpoint",
        }
    )
    out["target"] = _direction_to_target(out["turning_direction"])
    return out.sort_values("date").reset_index(drop=True)


def build_turning_target_frame(
    df_market: pd.DataFrame,
    df_turning_points: pd.DataFrame,
    mode: TargetMode,
    window_radius_days: int = 8,
    window_distance_power: float = 1.5,
    segment_center_power: float = 1.5,
    segment_direction_anchor: Literal["current_turn", "next_turn"] = "current_turn",
    include_last_open_segment: bool = True,
    segment_open_tail_direction_mode: Literal["last_turn", "endpoint_sign"] = "last_turn",
    segment_open_tail_min_move_pct: float = 0.0,
    min_weight: float = 0.05,
    use_amplitude_weight: bool = True,
    use_numba: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper around all target modes."""
    mode = str(mode)
    if mode == "point_only":
        return build_point_only_targets(
            df_turning_points=df_turning_points,
            use_amplitude_weight=use_amplitude_weight,
        )
    if mode == "window_kernel":
        return build_window_kernel_targets(
            df_market=df_market,
            df_turning_points=df_turning_points,
            radius_days=int(window_radius_days),
            distance_power=float(window_distance_power),
            min_weight=float(min_weight),
            use_amplitude_weight=use_amplitude_weight,
            use_numba=bool(use_numba),
        )
    if mode == "segment_midpoint":
        return build_segment_midpoint_targets(
            df_market=df_market,
            df_turning_points=df_turning_points,
            center_power=float(segment_center_power),
            segment_direction_anchor=segment_direction_anchor,
            include_last_open_segment=bool(include_last_open_segment),
            open_tail_direction_mode=segment_open_tail_direction_mode,
            open_tail_min_move_pct=float(segment_open_tail_min_move_pct),
            min_weight=float(min_weight),
            use_amplitude_weight=use_amplitude_weight,
            use_numba=bool(use_numba),
        )
    raise ValueError("mode must be one of: point_only, window_kernel, segment_midpoint")


def merge_features_with_turning_target(
    df_features: pd.DataFrame,
    df_target: pd.DataFrame,
    df_market_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge feature matrix with turning target frame by date.

    Result keeps only labeled rows (binary UP/DOWN target).
    """
    if "date" not in df_features.columns:
        raise ValueError("df_features must contain 'date'")
    if "date" not in df_target.columns or "target" not in df_target.columns:
        raise ValueError("df_target must contain 'date' and 'target'")

    left = df_features.copy()
    left["date"] = pd.to_datetime(left["date"])

    right = df_target.copy()
    right["date"] = pd.to_datetime(right["date"])

    out = pd.merge(left, right, on="date", how="inner")

    if df_market_close is not None and "close" in df_market_close.columns:
        close_df = df_market_close[["date", "close"]].copy()
        close_df["date"] = pd.to_datetime(close_df["date"])
        out = pd.merge(out, close_df, on="date", how="left")

    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["target"] = pd.to_numeric(out["target"], errors="coerce").astype(np.int32)
    out["sample_weight"] = pd.to_numeric(out.get("sample_weight", 1.0), errors="coerce").fillna(1.0)
    return out
