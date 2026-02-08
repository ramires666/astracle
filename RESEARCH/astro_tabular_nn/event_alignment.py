"""Event-alignment helpers for price-vs-regime diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RESEARCH2.Moon_cycles.turning_points import TurningPointLabelConfig, label_turning_points


def build_switch_events(dates: Iterable[Any], labels: np.ndarray) -> pd.DataFrame:
    """Return dates where label switches happen."""
    d = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    y = np.asarray(labels, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    if len(y) < 2:
        return pd.DataFrame(columns=["date", "new_regime", "prev_regime"])
    for i in range(1, len(y)):
        prev_label = int(y[i - 1])
        cur_label = int(y[i])
        if cur_label != prev_label:
            rows.append({"date": pd.to_datetime(d.iloc[i]), "new_regime": cur_label, "prev_regime": prev_label})
    return pd.DataFrame(rows)


def _segments_from_labels(labels: np.ndarray) -> list[tuple[int, int, int]]:
    y = np.asarray(labels, dtype=np.int32)
    if y.size == 0:
        return []
    segs: list[tuple[int, int, int]] = []
    start = 0
    current = int(y[0])
    for i in range(1, len(y)):
        if int(y[i]) != current:
            segs.append((start, i, current))
            start = i
            current = int(y[i])
    segs.append((start, len(y), current))
    return segs


def _stabilize_short_segments(labels: np.ndarray, min_segment_days: int) -> np.ndarray:
    """Merge very short segments to neighboring regime."""
    y = np.asarray(labels, dtype=np.int32).copy()
    m = int(max(1, min_segment_days))
    if y.size == 0 or m <= 1:
        return y

    for _ in range(10):
        changed = False
        segs = _segments_from_labels(y)
        if len(segs) <= 1:
            break
        for i, (s, e, _) in enumerate(segs):
            if int(e - s) >= m:
                continue
            if i == 0 and len(segs) > 1:
                y[s:e] = int(segs[i + 1][2])
                changed = True
            elif i == len(segs) - 1 and len(segs) > 1:
                y[s:e] = int(segs[i - 1][2])
                changed = True
            elif 0 < i < len(segs) - 1:
                prev_len = int(segs[i - 1][1] - segs[i - 1][0])
                next_len = int(segs[i + 1][1] - segs[i + 1][0])
                pick = int(segs[i - 1][2]) if prev_len >= next_len else int(segs[i + 1][2])
                y[s:e] = pick
                changed = True
        if not changed:
            break
    return y


def _build_regime_labels_from_signal(
    signal: np.ndarray,
    enter_up: float,
    enter_down: float,
    smooth_span_days: int,
    min_segment_days: int,
) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return np.array([], dtype=np.int32)

    if int(max(1, smooth_span_days)) > 1:
        smooth = (
            pd.Series(sig, dtype=float)
            .ewm(span=int(max(1, smooth_span_days)), adjust=False)
            .mean()
            .to_numpy(dtype=float)
        )
    else:
        smooth = sig.copy()

    up = float(enter_up)
    down = float(enter_down)
    if down > up:
        down, up = up, down

    regime = np.zeros(len(smooth), dtype=np.int32)
    mid = 0.5 * (up + down)
    regime[0] = 1 if float(smooth[0]) >= mid else 0

    for i in range(1, len(smooth)):
        v = float(smooth[i])
        if v >= up:
            regime[i] = 1
        elif v <= down:
            regime[i] = 0
        else:
            regime[i] = regime[i - 1]

    return _stabilize_short_segments(regime, min_segment_days=int(min_segment_days))


def build_pred_regime_labels_from_frame(
    test_frame: pd.DataFrame,
    mode: str,
    smooth_span_days: int,
    enter_up: float,
    enter_down: float,
    min_segment_days: int,
    center_threshold: float,
    center_delta_up: float,
    center_delta_down: float,
) -> np.ndarray:
    """Build predicted regime labels from hard labels or probability signal."""
    if test_frame.empty:
        return np.array([], dtype=np.int32)

    m = str(mode).lower().strip()
    if m in {"hard_label", "hard", "pred_label"}:
        y = pd.to_numeric(test_frame["pred"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
        return _stabilize_short_segments(y, min_segment_days=int(min_segment_days))

    proba = pd.to_numeric(test_frame["proba_up"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    if m in {"proba_threshold_centered", "threshold_centered", "centered_proba"}:
        c = float(center_threshold)
        up = min(0.999, max(0.001, c + max(0.0, float(center_delta_up))))
        down = min(0.999, max(0.001, c - max(0.0, float(center_delta_down))))
        return _build_regime_labels_from_signal(
            signal=proba,
            enter_up=up,
            enter_down=down,
            smooth_span_days=int(smooth_span_days),
            min_segment_days=int(min_segment_days),
        )
    if m in {"proba_smooth_regime", "proba", "probability"}:
        return _build_regime_labels_from_signal(
            signal=proba,
            enter_up=float(enter_up),
            enter_down=float(enter_down),
            smooth_span_days=int(smooth_span_days),
            min_segment_days=int(min_segment_days),
        )
    raise ValueError(f"Unsupported pred mode: {mode}")


def match_events_by_window(true_events: pd.DataFrame, pred_events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Greedy one-to-one matching by min abs lag within ±window_days."""
    if true_events.empty or pred_events.empty:
        return pd.DataFrame(columns=["true_date", "pred_date", "new_regime", "lag_days", "abs_lag"])

    used_pred: set[int] = set()
    rows: list[dict[str, Any]] = []
    for _, true_row in true_events.iterrows():
        true_date = pd.to_datetime(true_row["date"])
        true_regime = int(true_row["new_regime"])
        best_key: Optional[tuple[int, int, int]] = None
        best_data: Optional[dict[str, Any]] = None

        for pred_idx, pred_row in pred_events.iterrows():
            idx = int(pred_idx)
            if idx in used_pred:
                continue
            if int(pred_row["new_regime"]) != true_regime:
                continue

            pred_date = pd.to_datetime(pred_row["date"])
            lag = int((pred_date - true_date).days)
            abs_lag = abs(lag)
            if abs_lag > int(window_days):
                continue

            key = (abs_lag, 0 if lag <= 0 else 1, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_data = {"pred_idx": idx, "pred_date": pred_date, "lag_days": lag, "abs_lag": abs_lag}

        if best_data is not None:
            used_pred.add(int(best_data["pred_idx"]))
            rows.append(
                {
                    "true_date": true_date,
                    "pred_date": best_data["pred_date"],
                    "new_regime": true_regime,
                    "lag_days": best_data["lag_days"],
                    "abs_lag": best_data["abs_lag"],
                }
            )

    return pd.DataFrame(rows)


def compute_event_metrics(true_events: pd.DataFrame, pred_events: pd.DataFrame, matches: pd.DataFrame) -> Dict[str, float]:
    """Compute compact event metrics used in chart title and summary."""
    n_true = int(len(true_events))
    n_pred = int(len(pred_events))
    n_match = int(len(matches))
    recall_true = float(n_match / n_true) if n_true > 0 else float("nan")
    precision_pred = float(n_match / n_pred) if n_pred > 0 else float("nan")
    mean_abs_lag = float(pd.to_numeric(matches["abs_lag"], errors="coerce").dropna().mean()) if n_match > 0 else float("nan")
    return {
        "n_true_events": float(n_true),
        "n_pred_events": float(n_pred),
        "n_matched": float(n_match),
        "recall_true": recall_true,
        "precision_pred": precision_pred,
        "mean_abs_lag_days": mean_abs_lag,
    }


def build_global_true_events(
    df_market_ref: pd.DataFrame,
    horizon_days: int,
    up_move_pct: float,
    down_move_pct: float,
    cluster_gap_days: int,
    min_turn_gap_days: int,
    past_horizon_days: int,
    past_up_move_pct: float,
    past_down_move_pct: float,
    tail_direction_mode: str,
    tail_min_move_pct: float,
) -> pd.DataFrame:
    """Build global TP-based true events (same idea as old notebook)."""
    tp_cfg = TurningPointLabelConfig(
        horizon_days=int(horizon_days),
        up_move_pct=float(up_move_pct),
        down_move_pct=float(down_move_pct),
        cluster_gap_days=int(cluster_gap_days),
        min_turn_gap_days=int(min_turn_gap_days),
        past_horizon_days=int(past_horizon_days),
        past_up_move_pct=float(past_up_move_pct),
        past_down_move_pct=float(past_down_move_pct),
        tail_direction_mode=str(tail_direction_mode),
        tail_min_move_pct=float(tail_min_move_pct),
    )
    _, df_turns, _ = label_turning_points(df_market=df_market_ref, cfg=tp_cfg)
    out = df_turns.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["new_regime"] = (pd.to_numeric(out["turning_direction"], errors="coerce").fillna(0).astype(int) == 1).astype(int)
    return out[["date", "new_regime"]].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def plot_event_alignment(
    test_frame: pd.DataFrame,
    true_events: pd.DataFrame,
    pred_labels: np.ndarray,
    matches: pd.DataFrame,
    title_prefix: str,
    window_days: int,
    true_mode: str,
    out_png: Path,
) -> None:
    """Render old-notebook-like chart and save as PNG."""
    frame = test_frame.copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    prices = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    if len(frame) == 0:
        raise ValueError("test_frame is empty")

    p_min = float(np.nanmin(prices))
    p_max = float(np.nanmax(prices))
    margin = (p_max - p_min) * 0.05 if p_max > p_min else 1.0
    fill_min = p_min - margin
    fill_max = p_max + margin

    pred_events = build_switch_events(frame["date"], pred_labels)
    metrics = compute_event_metrics(true_events=true_events, pred_events=pred_events, matches=matches)

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(frame["date"], prices, color="#1f77b4", linewidth=1.5, label="Price")
    ax.fill_between(frame["date"], fill_min, fill_max, where=(pred_labels == 1), color="green", alpha=0.20, step="mid", label="Pred UP")
    ax.fill_between(frame["date"], fill_min, fill_max, where=(pred_labels == 0), color="red", alpha=0.20, step="mid", label="Pred DOWN")

    for _, row in true_events.iterrows():
        color = "#0b6e0b" if int(row["new_regime"]) == 1 else "#b30000"
        ax.axvline(pd.to_datetime(row["date"]), color=color, linestyle="--", alpha=0.6, linewidth=1.2)
    for _, row in matches.iterrows():
        ax.plot(pd.to_datetime(row["pred_date"]), fill_max, marker="o", markersize=4, color="black")

    ax.set_ylim(fill_min, fill_max)
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_title(
        f"{title_prefix} | true_mode={true_mode} | event_window=±{window_days}d | "
        f"recall_true={metrics['recall_true']:.3f} precision_pred={metrics['precision_pred']:.3f} "
        f"mean_abs_lag={metrics['mean_abs_lag_days']:.2f}d"
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

