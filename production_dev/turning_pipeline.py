"""Turning-grid helpers for production training/inference/backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from RESEARCH.data_loader import load_market_data
from RESEARCH.model_training import check_cuda_available
from RESEARCH2.Moon_cycles.eval_utils import compute_binary_metrics
from RESEARCH2.Moon_cycles.splits import make_classic_split
from RESEARCH2.Moon_cycles.turning_astro_features import (
    TurningAstroFeatureConfig,
    build_turning_astro_feature_set,
)
from RESEARCH2.Moon_cycles.turning_points import TurningPointLabelConfig, label_turning_points
from RESEARCH2.Moon_cycles.turning_targets import (
    build_turning_target_frame,
    merge_features_with_turning_target,
)
from src.models.xgb import XGBBaseline


@dataclass(frozen=True)
class TurningTrainArtifacts:
    model: XGBBaseline
    feature_names: List[str]
    threshold: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_profit_y: float
    test_profit_y: float
    val_profit_y_obj: float
    test_profit_y_obj: float
    train_rows: int
    val_rows: int
    test_rows: int


def load_market_frame(
    data_start: str = "2017-11-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load and normalize market daily frame used by turning-grid training."""
    df = load_market_data()
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    df = df.reset_index(drop=True)

    if data_start:
        df = df[df["date"] >= pd.to_datetime(data_start)].copy()
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)].copy()

    return df.reset_index(drop=True)


def _close_map_with_next_ret(df_market: pd.DataFrame) -> pd.DataFrame:
    """Build date->close->next_ret frame."""
    close_map = df_market[["date", "close"]].copy()
    close_map["next_close"] = close_map["close"].shift(-1)
    close_map["next_ret"] = close_map["next_close"] / close_map["close"] - 1.0
    return close_map


def build_artifact_config_from_checkpoint_row(row: pd.Series) -> Dict[str, Any]:
    """Convert one turning-grid checkpoint row into portable artifact config."""
    target_mode = str(row["target_mode"])
    target_cfg: Dict[str, Any] = {
        "mode": target_mode,
        "min_weight": float(row["target_min_weight"]),
        "use_amplitude_weight": bool(row["target_use_amplitude_weight"]),
    }
    if target_mode == "window_kernel":
        target_cfg.update(
            {
                "window_radius_days": int(row["target_window_radius_days"]),
                "window_distance_power": float(row["target_window_distance_power"]),
            }
        )
    elif target_mode == "segment_midpoint":
        target_cfg.update(
            {
                "segment_center_power": float(row["target_segment_center_power"]),
                "segment_direction_anchor": str(row["target_segment_direction_anchor"]),
                "include_last_open_segment": bool(row["target_include_last_open_segment"]),
                "segment_open_tail_direction_mode": str(row["target_segment_open_tail_direction_mode"]),
                "segment_open_tail_min_move_pct": float(row["target_segment_open_tail_min_move_pct"]),
            }
        )
    else:
        raise ValueError(f"Unsupported target_mode={target_mode}")

    birth_dt_utc = str(row.get("birth_dt_utc", "2009-10-10T18:15:05Z"))
    birth_date = birth_dt_utc[:10] if len(birth_dt_utc) >= 10 else birth_dt_utc
    threshold = pd.to_numeric(row.get("threshold", np.nan), errors="coerce")
    if not np.isfinite(threshold):
        threshold = 0.5

    cfg = {
        # Family markers used by production code to branch logic.
        "model_family": "turning_massive_label_grid",
        "feature_pipeline": "turning_astro_v1",
        # UI compatibility keys.
        "birth_dt_utc": birth_dt_utc,
        "birth_date": birth_date,
        "coord_mode": str(row.get("feature_coord_mode", "both")),
        "orb_mult": float(row.get("feature_orb_mult", 0.10)),
        # Core tuned configs.
        "feature_cfg": {
            "coord_mode": str(row.get("feature_coord_mode", "both")),
            "orb_mult": float(row.get("feature_orb_mult", 0.10)),
            "include_pair_aspects": True,
            "include_phases": True,
            "include_transit_aspects": True,
            "add_trig_for_longitudes": True,
            "add_trig_for_moon_phase": True,
            "add_trig_for_elongations": True,
        },
        "label_cfg": {
            "up_move_pct": float(row["label_up_move_pct"]),
            "down_move_pct": float(row["label_down_move_pct"]),
            "cluster_gap_days": int(row["label_cluster_gap_days"]),
            "min_turn_gap_days": int(row["label_min_turn_gap_days"]),
            "past_horizon_days": int(row["label_past_horizon_days"]),
            "past_up_move_pct": float(row["label_past_up_move_pct"]),
            "past_down_move_pct": float(row["label_past_down_move_pct"]),
        },
        "target_cfg": target_cfg,
        "model_cfg": {
            "n_estimators": int(row["model_n_estimators"]),
            "max_depth": int(row["model_max_depth"]),
            "learning_rate": float(row["model_learning_rate"]),
            "subsample": float(row["model_subsample"]),
            "colsample_bytree": float(row["model_colsample_bytree"]),
            "early_stopping_rounds": int(row["model_early_stopping_rounds"]),
        },
        "horizon_days_fixed": int(row.get("horizon_days_fixed", 10)),
        "tail_direction_mode_fixed": str(row.get("tail_direction_mode_fixed", "endpoint_sign")),
        "tail_min_move_pct_fixed": float(row.get("tail_min_move_pct_fixed", 0.0)),
        # Split/threshold defaults used by cache/training.
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "decision_threshold": float(threshold),
    }
    return cfg


def build_turning_dataset_from_artifact_config(
    df_market: pd.DataFrame,
    config: Dict[str, Any],
    feature_names_hint: List[str] | None = None,
    cache_namespace: str = "research2_turning_grid",
    use_cache: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build turning dataset exactly as in massive turning-grid research notebook."""
    feature_cfg = dict(config.get("feature_cfg", {}))
    astro_cfg = TurningAstroFeatureConfig(
        coord_mode=str(feature_cfg.get("coord_mode", config.get("coord_mode", "both"))),
        orb_mult=float(feature_cfg.get("orb_mult", config.get("orb_mult", 0.10))),
        include_pair_aspects=bool(feature_cfg.get("include_pair_aspects", True)),
        include_phases=bool(feature_cfg.get("include_phases", True)),
        include_transit_aspects=bool(feature_cfg.get("include_transit_aspects", True)),
        add_trig_for_longitudes=bool(feature_cfg.get("add_trig_for_longitudes", True)),
        add_trig_for_moon_phase=bool(feature_cfg.get("add_trig_for_moon_phase", True)),
        add_trig_for_elongations=bool(feature_cfg.get("add_trig_for_elongations", True)),
    )
    birth_dt_utc = str(config.get("birth_dt_utc", "2009-10-10T18:15:05Z"))
    df_features = build_turning_astro_feature_set(
        df_market=df_market,
        birth_dt_utc=birth_dt_utc,
        cfg=astro_cfg,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        verbose=verbose,
        progress=False,
    )

    label_cfg = dict(config.get("label_cfg", {}))
    turn_cfg = TurningPointLabelConfig(
        horizon_days=int(config.get("horizon_days_fixed", 10)),
        up_move_pct=float(label_cfg["up_move_pct"]),
        down_move_pct=float(label_cfg["down_move_pct"]),
        cluster_gap_days=int(label_cfg["cluster_gap_days"]),
        min_turn_gap_days=int(label_cfg["min_turn_gap_days"]),
        past_horizon_days=int(label_cfg["past_horizon_days"]),
        past_up_move_pct=float(label_cfg["past_up_move_pct"]),
        past_down_move_pct=float(label_cfg["past_down_move_pct"]),
        tail_direction_mode=str(config.get("tail_direction_mode_fixed", "endpoint_sign")),
        tail_min_move_pct=float(config.get("tail_min_move_pct_fixed", 0.0)),
    )
    _, df_turns, _ = label_turning_points(df_market=df_market, cfg=turn_cfg)

    target_cfg = dict(config.get("target_cfg", {}))
    mode = str(target_cfg["mode"])
    if mode == "window_kernel":
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
    elif mode == "segment_midpoint":
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
    else:
        raise ValueError(f"Unsupported target mode: {mode}")

    close_map = _close_map_with_next_ret(df_market)
    df_dataset = merge_features_with_turning_target(
        df_features=df_features,
        df_target=df_target,
        df_market_close=df_market[["date", "close"]],
    )
    df_dataset = pd.merge(df_dataset, close_map[["date", "next_ret"]], on="date", how="left")
    df_dataset = (
        df_dataset
        .dropna(subset=["next_ret", "target", "sample_weight", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    feature_cols = [
        c for c in df_dataset.columns
        if c not in {
            "date", "target", "close", "next_ret",
            "turning_direction", "sample_weight", "target_mode",
            "event_index", "segment_index",
        }
    ]

    if feature_names_hint:
        missing = [c for c in feature_names_hint if c not in df_dataset.columns]
        if missing:
            zero_block = pd.DataFrame(0.0, index=df_dataset.index, columns=missing)
            df_dataset = pd.concat([df_dataset, zero_block], axis=1)
        feature_cols = list(feature_names_hint)

    return df_dataset, feature_cols


def _safe_predict_proba_up(model: XGBBaseline, X: np.ndarray) -> np.ndarray:
    """Predict P(UP) safely with CPU fallback for GPU model/data mismatch."""
    const_cls = getattr(model, "constant_class", None)
    if const_cls is not None:
        c = int(const_cls)
        return np.full(X.shape[0], 1.0 if c == 1 else 0.0, dtype=float)

    Xs = model.scaler.transform(X)
    booster = None
    restore_device = None
    try:
        booster = model.model.get_booster()
        restore_device = str(getattr(model, "device", "cpu"))
        booster.set_param({"device": "cpu"})
    except Exception:
        booster = None

    try:
        proba_up = model.model.predict_proba(Xs)[:, 1]
    finally:
        if booster is not None and restore_device and restore_device.startswith("cuda"):
            try:
                booster.set_param({"device": restore_device})
            except Exception:
                pass

    return np.asarray(proba_up, dtype=float)


def _profit_y(y_pred: np.ndarray, next_ret: np.ndarray) -> float:
    p = np.where(np.asarray(y_pred, dtype=np.int32) == 1, 1.0, -1.0)
    y = np.asarray(next_ret, dtype=float)
    return float(np.mean(p * y)) if len(y) > 0 else 0.0


def _weighted_move_vector(
    next_ret: np.ndarray,
    sample_weight: np.ndarray,
    power: float = 1.5,
    clip_q: float = 0.98,
) -> np.ndarray:
    base = np.abs(np.asarray(next_ret, dtype=float))
    if base.size == 0:
        return np.array([], dtype=float)
    cap = float(np.quantile(base, clip_q))
    if not np.isfinite(cap) or cap <= 0.0:
        cap = float(np.nanmax(base)) if np.isfinite(np.nanmax(base)) and np.nanmax(base) > 0 else 1.0
    move_part = np.clip(base / cap, 0.0, 1.0) ** float(power)
    w = move_part * np.asarray(sample_weight, dtype=float)
    return np.maximum(w, 1e-8)


def _profit_y_obj(y_pred: np.ndarray, next_ret: np.ndarray, sample_weight: np.ndarray) -> float:
    p = np.where(np.asarray(y_pred, dtype=np.int32) == 1, 1.0, -1.0)
    y = np.asarray(next_ret, dtype=float)
    w = _weighted_move_vector(next_ret=y, sample_weight=sample_weight)
    return float(np.sum(w * (p * y)) / np.sum(w))


def train_turning_split_model(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    config: Dict[str, Any],
    seed: int = 42,
) -> TurningTrainArtifacts:
    """Train split model (70/15/15) with tuned params from turning-grid."""
    split = make_classic_split(
        df_dataset,
        train_ratio=float(config.get("train_ratio", 0.70)),
        val_ratio=float(config.get("val_ratio", 0.15)),
    )
    train_df = df_dataset.iloc[split.train_idx].copy().reset_index(drop=True)
    val_df = df_dataset.iloc[split.val_idx].copy().reset_index(drop=True)
    test_df = df_dataset.iloc[split.test_idx].copy().reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.int32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.int32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.int32)

    sw_train_base = pd.to_numeric(train_df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    sw_val_base = pd.to_numeric(val_df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    sw_test_base = pd.to_numeric(test_df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)

    sw_train = sw_train_base * compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32)
    sw_val = sw_val_base * compute_sample_weight(class_weight="balanced", y=y_val).astype(np.float32)
    sw_test = sw_test_base * compute_sample_weight(class_weight="balanced", y=y_test).astype(np.float32)

    _, device = check_cuda_available()
    model_cfg = dict(config.get("model_cfg", {}))

    def _make_model(device_name: str) -> XGBBaseline:
        return XGBBaseline(
            n_classes=2,
            device=device_name,
            random_state=seed,
            early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=int(model_cfg["max_depth"]),
            learning_rate=float(model_cfg["learning_rate"]),
            subsample=float(model_cfg["subsample"]),
            colsample_bytree=float(model_cfg["colsample_bytree"]),
            tree_method="hist",
            eval_metric="logloss",
        )

    model = _make_model(str(device))
    try:
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_cols,
            sample_weight=sw_train,
            sample_weight_val=sw_val,
        )
    except Exception:
        model = _make_model("cpu")
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_cols,
            sample_weight=sw_train,
            sample_weight_val=sw_val,
        )

    threshold = float(config.get("decision_threshold", 0.5))
    p_val = _safe_predict_proba_up(model, X_val)
    p_test = _safe_predict_proba_up(model, X_test)

    pred_val = (p_val >= threshold).astype(np.int32)
    pred_test = (p_test >= threshold).astype(np.int32)

    val_metrics = compute_binary_metrics(y_val, pred_val)
    test_metrics = compute_binary_metrics(y_test, pred_test)

    ret_val = pd.to_numeric(val_df["next_ret"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ret_test = pd.to_numeric(test_df["next_ret"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    return TurningTrainArtifacts(
        model=model,
        feature_names=feature_cols,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_profit_y=_profit_y(pred_val, ret_val),
        test_profit_y=_profit_y(pred_test, ret_test),
        val_profit_y_obj=_profit_y_obj(pred_val, ret_val, sw_val),
        test_profit_y_obj=_profit_y_obj(pred_test, ret_test, sw_test),
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )


def train_turning_full_model(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    config: Dict[str, Any],
    seed: int = 42,
) -> XGBBaseline:
    """Train FULL model on all rows with turning-grid hyperparameters."""
    X_all = df_dataset[feature_cols].to_numpy(dtype=np.float32)
    y_all = df_dataset["target"].to_numpy(dtype=np.int32)
    sw_base = pd.to_numeric(df_dataset["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    sw_all = sw_base * compute_sample_weight(class_weight="balanced", y=y_all).astype(np.float32)

    _, device = check_cuda_available()
    model_cfg = dict(config.get("model_cfg", {}))
    model = XGBBaseline(
        n_classes=2,
        device=str(device),
        random_state=seed,
        early_stopping_rounds=None,
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=int(model_cfg["max_depth"]),
        learning_rate=float(model_cfg["learning_rate"]),
        subsample=float(model_cfg["subsample"]),
        colsample_bytree=float(model_cfg["colsample_bytree"]),
        tree_method="hist",
        eval_metric="logloss",
    )
    try:
        model.fit(
            X_train=X_all,
            y_train=y_all,
            X_val=None,
            y_val=None,
            feature_names=feature_cols,
            sample_weight=sw_all,
            sample_weight_val=None,
        )
    except Exception:
        model = XGBBaseline(
            n_classes=2,
            device="cpu",
            random_state=seed,
            early_stopping_rounds=None,
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=int(model_cfg["max_depth"]),
            learning_rate=float(model_cfg["learning_rate"]),
            subsample=float(model_cfg["subsample"]),
            colsample_bytree=float(model_cfg["colsample_bytree"]),
            tree_method="hist",
            eval_metric="logloss",
        )
        model.fit(
            X_train=X_all,
            y_train=y_all,
            X_val=None,
            y_val=None,
            feature_names=feature_cols,
            sample_weight=sw_all,
            sample_weight_val=None,
        )

    return model
