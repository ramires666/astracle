"""Segment-weighted directional objective ported from TP weighted notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentThresholdSearchResult:
    """Best threshold search result for segment-weighted binary objective."""

    best_threshold: float
    best_score: float
    best_idx: int
    scores: np.ndarray
    recall_min: np.ndarray
    recall_gap: np.ndarray
    accuracy: np.ndarray
    pred_up_share: np.ndarray
    weighted_hit_rate: np.ndarray
    weighted_majority_hit: np.ndarray


def _empty_segment_metrics() -> Dict[str, float]:
    return {
        "n_segments": 0.0,
        "n_segment_days": 0.0,
        "weighted_hit_rate": np.nan,
        "unweighted_hit_rate": np.nan,
        "weighted_majority_hit": np.nan,
        "unweighted_majority_hit": np.nan,
    }


def _coerce_binary_label(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return (numeric >= 0.5).astype(int)


def build_true_events_from_segment_index(
    frame: pd.DataFrame,
    segment_col: str = "segment_index",
    target_col: str = "target",
) -> pd.DataFrame:
    """Build event table (`date`, `new_regime`) from per-row segment labels."""
    req = {"date", segment_col, target_col}
    if frame.empty or not req.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "new_regime"])

    work = frame[["date", segment_col, target_col]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["date", "new_regime"])

    rows: list[dict[str, object]] = []
    for _, seg in work.groupby(segment_col, sort=False):
        if seg.empty:
            continue
        start_date = pd.to_datetime(seg["date"].iloc[0])
        regime = int(_coerce_binary_label(seg[target_col]).iloc[0])
        rows.append({"date": start_date, "new_regime": regime})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "new_regime"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def build_tp_segments_from_events(
    test_frame: pd.DataFrame,
    true_events: pd.DataFrame,
    gamma: float = 1.5,
    min_days: int = 5,
    include_open_tail: bool = True,
) -> pd.DataFrame:
    """Convert TP event dates into directed movement segments on one split."""
    if test_frame.empty or true_events.empty:
        return pd.DataFrame()

    frame = test_frame[["date", "close"]].copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    events = true_events[["date", "new_regime"]].copy()
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["new_regime"] = _coerce_binary_label(events["new_regime"])
    events = events.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if events.empty:
        return pd.DataFrame()

    dates = frame["date"].to_numpy()
    idx = np.searchsorted(dates, events["date"].to_numpy(), side="left")
    valid = (idx >= 0) & (idx < len(frame))
    events = events.loc[valid].reset_index(drop=True)
    idx = idx[valid]
    if len(events) == 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    m_days = int(max(1, min_days))

    for i in range(0, len(events) - 1):
        s_idx = int(idx[i])
        e_idx_excl = int(idx[i + 1])
        if e_idx_excl <= s_idx:
            continue
        n_days = int(e_idx_excl - s_idx)
        if n_days < m_days:
            continue

        start_price = float(frame.loc[s_idx, "close"])
        end_price = float(frame.loc[e_idx_excl - 1, "close"])
        if not (np.isfinite(start_price) and np.isfinite(end_price) and start_price > 0.0 and end_price > 0.0):
            continue

        signed_log_ret = float(np.log(end_price / start_price))
        amp = float(abs(signed_log_ret))
        weight = float(max(1e-12, amp) ** float(gamma))

        rows.append(
            {
                "segment_id": int(i),
                "start_idx": s_idx,
                "end_idx_excl": e_idx_excl,
                "n_days": n_days,
                "start_date": pd.to_datetime(frame.loc[s_idx, "date"]),
                "end_date": pd.to_datetime(frame.loc[e_idx_excl - 1, "date"]),
                "true_regime": int(events.loc[i, "new_regime"]),
                "start_price": start_price,
                "end_price": end_price,
                "signed_log_return": signed_log_ret,
                "amplitude": amp,
                "weight": weight,
            }
        )

    if bool(include_open_tail) and len(events) >= 1:
        s_idx = int(idx[-1])
        e_idx_excl = int(len(frame))
        if e_idx_excl > s_idx:
            n_days = int(e_idx_excl - s_idx)
            if n_days >= m_days:
                start_price = float(frame.loc[s_idx, "close"])
                end_price = float(frame.loc[e_idx_excl - 1, "close"])
                if np.isfinite(start_price) and np.isfinite(end_price) and start_price > 0.0 and end_price > 0.0:
                    signed_log_ret = float(np.log(end_price / start_price))
                    amp = float(abs(signed_log_ret))
                    weight = float(max(1e-12, amp) ** float(gamma))
                    rows.append(
                        {
                            "segment_id": int(len(rows)),
                            "start_idx": s_idx,
                            "end_idx_excl": e_idx_excl,
                            "n_days": n_days,
                            "start_date": pd.to_datetime(frame.loc[s_idx, "date"]),
                            "end_date": pd.to_datetime(frame.loc[e_idx_excl - 1, "date"]),
                            "true_regime": int(events.loc[len(events) - 1, "new_regime"]),
                            "start_price": start_price,
                            "end_price": end_price,
                            "signed_log_return": signed_log_ret,
                            "amplitude": amp,
                            "weight": weight,
                            "is_open_tail": True,
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "is_open_tail" not in out.columns:
        out["is_open_tail"] = False
    return out.sort_values(["start_idx", "end_idx_excl"]).reset_index(drop=True)


def build_tp_segments_from_frame(
    frame: pd.DataFrame,
    gamma: float = 1.5,
    min_days: int = 5,
    include_open_tail: bool = True,
) -> pd.DataFrame:
    """Build TP segments from split frame; prefers explicit `segment_index` when available."""
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame()

    if {"segment_index", "target"}.issubset(frame.columns):
        events = build_true_events_from_segment_index(frame=frame, segment_col="segment_index", target_col="target")
    elif "target" in frame.columns:
        work = frame[["date", "target"]].copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if work.empty:
            events = pd.DataFrame(columns=["date", "new_regime"])
        else:
            regime = _coerce_binary_label(work["target"]).to_numpy(dtype=np.int64)
            switch = np.ones(len(work), dtype=bool)
            switch[1:] = regime[1:] != regime[:-1]
            events = pd.DataFrame(
                {
                    "date": pd.to_datetime(work.loc[switch, "date"]).reset_index(drop=True),
                    "new_regime": regime[switch].astype(int),
                }
            )
    else:
        events = pd.DataFrame(columns=["date", "new_regime"])

    if events.empty:
        return pd.DataFrame()

    return build_tp_segments_from_events(
        test_frame=frame,
        true_events=events,
        gamma=float(gamma),
        min_days=int(min_days),
        include_open_tail=bool(include_open_tail),
    )


def score_predictions_on_tp_segments(
    pred_labels: np.ndarray,
    segments: pd.DataFrame,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """Compute segment-level and aggregate directional scores."""
    y = np.asarray(pred_labels, dtype=np.int32)
    if segments.empty or y.size == 0:
        return _empty_segment_metrics(), pd.DataFrame()

    det_rows: list[dict[str, object]] = []
    for _, seg in segments.iterrows():
        s = int(seg["start_idx"])
        e = int(seg["end_idx_excl"])
        if s < 0 or e > len(y) or e <= s:
            continue

        y_seg = y[s:e]
        true_reg = int(seg["true_regime"])
        hit_rate = float(np.mean(y_seg == true_reg))
        pred_majority = int(np.mean(y_seg) >= 0.5)
        majority_hit = float(pred_majority == true_reg)

        det_rows.append(
            {
                "segment_id": int(seg["segment_id"]),
                "start_date": pd.to_datetime(seg["start_date"]),
                "end_date": pd.to_datetime(seg["end_date"]),
                "n_days": int(seg["n_days"]),
                "true_regime": true_reg,
                "amplitude": float(seg["amplitude"]),
                "weight": float(seg["weight"]),
                "signed_log_return": float(seg["signed_log_return"]),
                "hit_rate": hit_rate,
                "pred_majority": pred_majority,
                "majority_hit": majority_hit,
                "is_open_tail": bool(seg.get("is_open_tail", False)),
            }
        )

    details = pd.DataFrame(det_rows)
    if details.empty:
        return _empty_segment_metrics(), details

    w = pd.to_numeric(details["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = np.maximum(w, 1e-12)

    hit = pd.to_numeric(details["hit_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    maj = pd.to_numeric(details["majority_hit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out = {
        "n_segments": float(len(details)),
        "n_segment_days": float(pd.to_numeric(details["n_days"], errors="coerce").fillna(0.0).sum()),
        "weighted_hit_rate": float(np.sum(w * hit) / np.sum(w)),
        "unweighted_hit_rate": float(np.mean(hit)),
        "weighted_majority_hit": float(np.sum(w * maj) / np.sum(w)),
        "unweighted_majority_hit": float(np.mean(maj)),
    }
    return out, details


def search_best_threshold_segment_weighted(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    segments: pd.DataFrame,
    gap_penalty: float,
    prior_penalty: float,
    segment_metric: str = "weighted_hit_rate",
) -> SegmentThresholdSearchResult:
    """Find threshold maximizing segment-weighted reward minus balance penalties."""
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probs must be 2D with at least 2 class columns")
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be 1D array")
    metric_name = str(segment_metric).strip().lower()
    if metric_name not in {"weighted_hit_rate", "weighted_majority_hit"}:
        raise ValueError(f"Unsupported segment_metric: {segment_metric}")

    p_up = np.asarray(probs[:, 1], dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    thr_grid = np.asarray(thresholds, dtype=np.float64)
    n = len(y)
    if n == 0:
        raise ValueError("y_true is empty")

    true_up = int(np.sum(y == 1))
    true_down = int(np.sum(y == 0))
    true_up_share = float(true_up / max(n, 1))

    scores = np.empty(len(thr_grid), dtype=np.float64)
    recall_min = np.empty(len(thr_grid), dtype=np.float64)
    recall_gap = np.empty(len(thr_grid), dtype=np.float64)
    accuracy = np.empty(len(thr_grid), dtype=np.float64)
    pred_up_share = np.empty(len(thr_grid), dtype=np.float64)
    weighted_hit_rate = np.empty(len(thr_grid), dtype=np.float64)
    weighted_majority_hit = np.empty(len(thr_grid), dtype=np.float64)

    for i, thr in enumerate(thr_grid.tolist()):
        pred = (p_up >= float(thr)).astype(np.int64)
        hit_up = int(np.sum((pred == 1) & (y == 1)))
        hit_down = int(np.sum((pred == 0) & (y == 0)))
        rec_up = float(hit_up / max(true_up, 1))
        rec_down = float(hit_down / max(true_down, 1))
        rec_min = rec_down if rec_down < rec_up else rec_up
        gap = abs(rec_down - rec_up)

        acc = float(np.mean(pred == y))
        pred_share = float(np.mean(pred == 1))
        prior_gap = abs(pred_share - true_up_share)

        seg_metrics, _ = score_predictions_on_tp_segments(pred_labels=pred, segments=segments)
        seg_hit = float(seg_metrics["weighted_hit_rate"])
        seg_maj = float(seg_metrics["weighted_majority_hit"])
        seg_base = seg_hit if metric_name == "weighted_hit_rate" else seg_maj
        if not np.isfinite(seg_base):
            seg_base = -1e9

        score = float(seg_base - float(gap_penalty) * gap - float(prior_penalty) * prior_gap)

        scores[i] = score
        recall_min[i] = rec_min
        recall_gap[i] = gap
        accuracy[i] = acc
        pred_up_share[i] = pred_share
        weighted_hit_rate[i] = seg_hit
        weighted_majority_hit[i] = seg_maj

    best_idx = int(np.argmax(scores))
    return SegmentThresholdSearchResult(
        best_threshold=float(thr_grid[best_idx]),
        best_score=float(scores[best_idx]),
        best_idx=best_idx,
        scores=scores,
        recall_min=recall_min,
        recall_gap=recall_gap,
        accuracy=accuracy,
        pred_up_share=pred_up_share,
        weighted_hit_rate=weighted_hit_rate,
        weighted_majority_hit=weighted_majority_hit,
    )
