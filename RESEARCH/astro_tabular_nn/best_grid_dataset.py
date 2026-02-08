"""Utilities for building the exact `grid_best` labeled dataset used in latest TP notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from RESEARCH.cache_utils import get_cache_path, load_cache, save_cache
from RESEARCH.config import cfg as project_cfg
from RESEARCH.data_loader import load_market_data
from RESEARCH2.Moon_cycles.turning_astro_features import TurningAstroFeatureConfig, build_turning_astro_feature_set
from RESEARCH2.Moon_cycles.turning_points import TurningPointLabelConfig, label_turning_points
from RESEARCH2.Moon_cycles.turning_targets import build_turning_target_frame, merge_features_with_turning_target

DEFAULT_RUN_TAG = "turning_massive_label_grid"
DEFAULT_DATA_START = "2017-11-01"
DEFAULT_FEATURE_CACHE_NAMESPACE = "research2_turning_grid"
DEFAULT_DATASET_CACHE_CATEGORY = "astro_tabular_nn_best_grid"
DEFAULT_DATASET_CACHE_NAME = "dataset"

_SUBJECT_CFG = getattr(project_cfg, "subject", {}) or {}


def _as_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(int(v))
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def resolve_checkpoint_path(run_tag: str = DEFAULT_RUN_TAG) -> Path:
    reports_dir = project_cfg.reports_dir if hasattr(project_cfg, "reports_dir") else Path("data/market/reports")
    path = Path(reports_dir) / f"{run_tag}_checkpoint.csv"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def sort_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same sorting as in the latest TP notebook."""
    if df.empty:
        return df

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
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v

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


def load_grid_best_row(checkpoint_path: Path) -> pd.Series:
    """Return top `grid_best` row from checkpoint using notebook ranking."""
    df = pd.read_csv(checkpoint_path)
    if df.empty:
        raise ValueError(f"Checkpoint is empty: {checkpoint_path}")
    sorted_df = sort_results_frame(df)
    return sorted_df.iloc[0].copy()


def build_market_and_close_map(data_start: str = DEFAULT_DATA_START) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load market series and create next-day return map."""
    df_market = load_market_data()
    df_market = df_market[df_market["date"] >= data_start].copy()
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market["close"] = pd.to_numeric(df_market["close"], errors="coerce")
    df_market = (
        df_market.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    close_map = df_market[["date", "close"]].copy()
    close_map["next_close"] = close_map["close"].shift(-1)
    close_map["next_ret"] = close_map["next_close"] / close_map["close"] - 1.0
    return df_market, close_map


def _extract_label_cfg(row: pd.Series) -> Dict[str, Any]:
    return {
        "up_move_pct": float(row["label_up_move_pct"]),
        "down_move_pct": float(row["label_down_move_pct"]),
        "cluster_gap_days": int(row["label_cluster_gap_days"]),
        "min_turn_gap_days": int(row["label_min_turn_gap_days"]),
        "past_horizon_days": int(row["label_past_horizon_days"]),
        "past_up_move_pct": float(row["label_past_up_move_pct"]),
        "past_down_move_pct": float(row["label_past_down_move_pct"]),
    }


def _extract_target_cfg(row: pd.Series) -> Dict[str, Any]:
    mode = str(row["target_mode"])
    cfg: Dict[str, Any] = {
        "mode": mode,
        "min_weight": float(row["target_min_weight"]),
        "use_amplitude_weight": _as_bool(row["target_use_amplitude_weight"]),
    }
    if mode == "window_kernel":
        cfg.update(
            {
                "window_radius_days": int(row["target_window_radius_days"]),
                "window_distance_power": float(row["target_window_distance_power"]),
            }
        )
    elif mode == "segment_midpoint":
        cfg.update(
            {
                "segment_center_power": float(row["target_segment_center_power"]),
                "segment_direction_anchor": str(row["target_segment_direction_anchor"]),
                "include_last_open_segment": _as_bool(row["target_include_last_open_segment"]),
                "segment_open_tail_direction_mode": str(row["target_segment_open_tail_direction_mode"]),
                "segment_open_tail_min_move_pct": float(row["target_segment_open_tail_min_move_pct"]),
            }
        )
    else:
        raise ValueError(f"Unsupported target_mode: {mode}")
    return cfg


def _dataset_cache_params(row: pd.Series, run_tag: str, data_start: str) -> Dict[str, Any]:
    return {
        "run_tag": str(run_tag),
        "data_start": str(data_start),
        "eval_id": int(row.get("eval_id", -1)),
        "target_mode": str(row["target_mode"]),
        "feature_coord_mode": str(row.get("feature_coord_mode", "both")),
        "feature_orb_mult": float(row.get("feature_orb_mult", 0.10)),
        "birth_dt_utc": str(row.get("birth_dt_utc", _SUBJECT_CFG.get("birth_dt_utc", ""))),
        "label_up_move_pct": float(row["label_up_move_pct"]),
        "label_down_move_pct": float(row["label_down_move_pct"]),
        "target_min_weight": float(row["target_min_weight"]),
        "threshold": float(row.get("threshold", 0.5)),
    }


def build_best_grid_labeled_dataset(
    run_tag: str = DEFAULT_RUN_TAG,
    data_start: str = DEFAULT_DATA_START,
    feature_cache_namespace: str = DEFAULT_FEATURE_CACHE_NAMESPACE,
    dataset_cache_category: str = DEFAULT_DATASET_CACHE_CATEGORY,
    dataset_cache_name: str = DEFAULT_DATASET_CACHE_NAME,
    use_cache: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Path, pd.Series]:
    """Build exact `grid_best` labeled dataset from latest TP checkpoint row."""
    checkpoint_path = resolve_checkpoint_path(run_tag=run_tag)
    best_row = load_grid_best_row(checkpoint_path=checkpoint_path)
    cache_params = _dataset_cache_params(best_row, run_tag=run_tag, data_start=data_start)
    dataset_path = get_cache_path(dataset_cache_category, dataset_cache_name, cache_params, "parquet")

    if use_cache:
        cached = load_cache(dataset_cache_category, dataset_cache_name, cache_params, verbose=verbose)
        if cached is not None:
            if verbose:
                print(f"[best-grid dataset] loaded cached dataset: {dataset_path}")
            return cached, dataset_path, best_row

    df_market, close_map = build_market_and_close_map(data_start=data_start)

    birth_dt_utc = str(best_row.get("birth_dt_utc", _SUBJECT_CFG.get("birth_dt_utc", "2009-10-10T18:15:05Z")))
    coord_mode = str(best_row.get("feature_coord_mode", "both"))
    orb_mult = float(best_row.get("feature_orb_mult", 0.10))

    astro_cfg = TurningAstroFeatureConfig(
        coord_mode=coord_mode,
        orb_mult=orb_mult,
        include_pair_aspects=True,
        include_phases=True,
        include_transit_aspects=True,
        add_trig_for_longitudes=True,
        add_trig_for_moon_phase=True,
        add_trig_for_elongations=True,
    )

    df_features = build_turning_astro_feature_set(
        df_market=df_market,
        birth_dt_utc=birth_dt_utc,
        cfg=astro_cfg,
        cache_namespace=feature_cache_namespace,
        use_cache=True,
        verbose=verbose,
        progress=bool(verbose),
    )

    label_cfg = _extract_label_cfg(best_row)
    turn_cfg = TurningPointLabelConfig(
        horizon_days=int(best_row.get("horizon_days_fixed", 10)),
        up_move_pct=float(label_cfg["up_move_pct"]),
        down_move_pct=float(label_cfg["down_move_pct"]),
        cluster_gap_days=int(label_cfg["cluster_gap_days"]),
        min_turn_gap_days=int(label_cfg["min_turn_gap_days"]),
        past_horizon_days=int(label_cfg["past_horizon_days"]),
        past_up_move_pct=float(label_cfg["past_up_move_pct"]),
        past_down_move_pct=float(label_cfg["past_down_move_pct"]),
        tail_direction_mode=str(best_row.get("tail_direction_mode_fixed", "endpoint_sign")),
        tail_min_move_pct=float(best_row.get("tail_min_move_pct_fixed", 0.0)),
    )
    _, df_turns, _ = label_turning_points(df_market=df_market, cfg=turn_cfg)

    target_cfg = _extract_target_cfg(best_row)
    if target_cfg["mode"] == "window_kernel":
        df_target = build_turning_target_frame(
            df_market=df_market,
            df_turning_points=df_turns,
            mode="window_kernel",
            window_radius_days=int(target_cfg["window_radius_days"]),
            window_distance_power=float(target_cfg["window_distance_power"]),
            min_weight=float(target_cfg["min_weight"]),
            use_amplitude_weight=bool(target_cfg["use_amplitude_weight"]),
            use_numba=True,
        )
    else:
        df_target = build_turning_target_frame(
            df_market=df_market,
            df_turning_points=df_turns,
            mode="segment_midpoint",
            segment_center_power=float(target_cfg["segment_center_power"]),
            segment_direction_anchor=str(target_cfg["segment_direction_anchor"]),
            include_last_open_segment=bool(target_cfg["include_last_open_segment"]),
            segment_open_tail_direction_mode=str(target_cfg["segment_open_tail_direction_mode"]),
            segment_open_tail_min_move_pct=float(target_cfg["segment_open_tail_min_move_pct"]),
            min_weight=float(target_cfg["min_weight"]),
            use_amplitude_weight=bool(target_cfg["use_amplitude_weight"]),
            use_numba=True,
        )

    df_dataset = merge_features_with_turning_target(
        df_features=df_features,
        df_target=df_target,
        df_market_close=df_market[["date", "close"]],
    )
    df_dataset = pd.merge(df_dataset, close_map[["date", "next_ret"]], on="date", how="left")
    df_dataset = (
        df_dataset.dropna(subset=["next_ret", "target", "sample_weight", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    save_cache(df_dataset, dataset_cache_category, dataset_cache_name, cache_params, verbose=verbose)
    return df_dataset, dataset_path, best_row


def ensure_best_grid_dataset_path(
    run_tag: str = DEFAULT_RUN_TAG,
    data_start: str = DEFAULT_DATA_START,
    feature_cache_namespace: str = DEFAULT_FEATURE_CACHE_NAMESPACE,
    dataset_cache_category: str = DEFAULT_DATASET_CACHE_CATEGORY,
    dataset_cache_name: str = DEFAULT_DATASET_CACHE_NAME,
    use_cache: bool = True,
    verbose: bool = True,
) -> Path:
    """Build (or load) best-grid dataset and return parquet path."""
    _, path, _ = build_best_grid_labeled_dataset(
        run_tag=run_tag,
        data_start=data_start,
        feature_cache_namespace=feature_cache_namespace,
        dataset_cache_category=dataset_cache_category,
        dataset_cache_name=dataset_cache_name,
        use_cache=use_cache,
        verbose=verbose,
    )
    return path
