"""
Training step for astro_xgb pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight

from src.models.xgb import XGBBaseline
from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext
from src.pipeline.astro_xgb.metrics import calc_metrics, classification_report_dict


def _detect_device() -> str:
    try:
        import xgboost as xgb

        info = xgb.build_info()
        use_cuda = bool(info.get("USE_CUDA", False))
        return "cuda" if use_cuda else "cpu"
    except Exception:
        return "cpu"


def _threshold_tune(
    proba_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "bal_acc",
) -> Tuple[float, float]:
    """
    Tune threshold on validation set for binary classification.
    Returns best threshold and best score.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (proba_val >= t).astype(np.int32)
        m = calc_metrics(y_val, pred, labels=[0, 1])
        score = m.get(metric, m["bal_acc"])
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t, best_score


def _extract_tag(dataset_path: Path) -> str:
    stem = Path(dataset_path).stem
    return stem.replace(f"{stem.split('_dataset_', 1)[0]}_dataset_", "")


def run_train_step(
    ctx: PipelineContext,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    dataset_path: Path,
    force: bool = False,
    threshold_tuning: bool = True,
) -> Path:
    """
    Train XGBoost model and save artifact. Returns model path.
    """
    print("[TRAIN] Starting training step...")
    train_cfg = ctx.cfg_train.get("training", {})
    two_stage = bool(train_cfg.get("two_stage", True))
    tune_enabled = bool(train_cfg.get("tune_enabled", False))
    tune_n_iter = int(train_cfg.get("tune_n_iter", 25))
    tune_cv_splits = int(train_cfg.get("tune_cv_splits", 5))
    tune_scoring = str(train_cfg.get("tune_scoring", "f1_macro"))
    tune_verbose_per_fold = bool(train_cfg.get("tune_verbose_per_fold", False))
    tune_show_accuracy = bool(train_cfg.get("tune_show_accuracy", True))
    tune_use_small = bool(train_cfg.get("tune_use_small", True))
    if tune_use_small:
        tune_param_dist = train_cfg.get("tune_param_dist_small") or None
        print("[TRAIN] Tuning grid: SMALL")
    else:
        tune_param_dist = train_cfg.get("tune_param_dist_full") or None
        print("[TRAIN] Tuning grid: FULL")
    xgb_params = dict(train_cfg.get("xgb", {}) or {})

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    feature_cols = [c for c in df_train.columns if c not in ["date", "target"]]
    print(f"[TRAIN] Features: {len(feature_cols)}")
    print(f"[TRAIN] Train rows: {len(df_train)} | Val rows: {len(df_val)} | Test rows: {len(df_test)}")
    if len(df_train) > 0:
        dist = df_train["target"].value_counts(normalize=True).sort_index() * 100
        print("[TRAIN] Train class share (%):", {int(k): round(v, 2) for k, v in dist.items()})
    X_train = df_train[feature_cols].to_numpy(dtype=np.float64)
    y_train = df_train["target"].to_numpy(dtype=np.int32)
    X_val = df_val[feature_cols].to_numpy(dtype=np.float64)
    y_val = df_val["target"].to_numpy(dtype=np.int32)
    X_test = df_test[feature_cols].to_numpy(dtype=np.float64)
    y_test = df_test["target"].to_numpy(dtype=np.int32)

    labels = sorted(np.unique(np.concatenate([y_train, y_val, y_test])).tolist())
    n_classes = len(labels)
    is_binary = n_classes == 2
    if is_binary and two_stage:
        two_stage = False

    device = _detect_device()
    model_dir = ctx.root / "models_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = _extract_tag(dataset_path)
    out_path = model_dir / f"xgb_astro_{tag}.joblib"

    params = {
        "two_stage": bool(two_stage),
        "tune_enabled": bool(tune_enabled),
        "tune_n_iter": tune_n_iter,
        "tune_cv_splits": tune_cv_splits,
        "tune_scoring": tune_scoring,
        "xgb_params": xgb_params,
        "device": device,
        "n_classes": n_classes,
        "threshold_tuning": bool(threshold_tuning),
    }
    inputs = [Path(train_path), Path(val_path), Path(test_path)]

    if not force and is_cache_valid(out_path, params=params, inputs=inputs, step="train"):
        print("[TRAIN] Using cached model:", out_path)
        return out_path

    if two_stage:
        print("[TRAIN] Mode: TWO_STAGE (MOVE/NO_MOVE -> UP/DOWN)")
        # Stage 1: MOVE vs NO_MOVE (1 = move, 0 = no-move)
        y_train_move = (y_train != 1).astype(np.int32)
        y_val_move = (y_val != 1).astype(np.int32)
        y_test_move = (y_test != 1).astype(np.int32)

        w_train_move = compute_sample_weight(class_weight="balanced", y=y_train_move)
        w_val_move = compute_sample_weight(class_weight="balanced", y=y_val_move)

        model_move = XGBBaseline(
            n_classes=2,
            device=device,
            random_state=42,
            **xgb_params,
        )
        if tune_enabled:
            print("[TRAIN] Tuning enabled:",
                  f"n_iter={tune_n_iter}, cv_splits={tune_cv_splits}, scoring={tune_scoring}")
            print("[TRAIN] Tuning MOVE model...")
            best_params = model_move.tune(
                X_train,
                y_train_move,
                param_dist=tune_param_dist,
                n_iter=tune_n_iter,
                cv_splits=tune_cv_splits,
                scoring=tune_scoring,
                scale=True,
                sample_weight=w_train_move,
                verbose_per_fold=tune_verbose_per_fold,
                show_accuracy=tune_show_accuracy,
            )
            print("[TRAIN] MOVE best params:", best_params)
        else:
            model_move.fit(
                X_train,
                y_train_move,
                X_val=X_val,
                y_val=y_val_move,
                feature_names=feature_cols,
                sample_weight=w_train_move,
                sample_weight_val=w_val_move,
            )

        # Stage 2: UP vs DOWN on MOVE rows only
        mask_train_dir = y_train != 1
        mask_val_dir = y_val != 1
        X_train_dir = X_train[mask_train_dir]
        y_train_dir = (y_train[mask_train_dir] == 2).astype(np.int32)
        X_val_dir = X_val[mask_val_dir]
        y_val_dir = (y_val[mask_val_dir] == 2).astype(np.int32)

        w_train_dir = compute_sample_weight(class_weight="balanced", y=y_train_dir)
        w_val_dir = compute_sample_weight(class_weight="balanced", y=y_val_dir)

        model_dir_stage = XGBBaseline(
            n_classes=2,
            device=device,
            random_state=42,
            **xgb_params,
        )
        if tune_enabled:
            print("[TRAIN] Tuning enabled:",
                  f"n_iter={tune_n_iter}, cv_splits={tune_cv_splits}, scoring={tune_scoring}")
            print("[TRAIN] Tuning DIR model...")
            best_params = model_dir_stage.tune(
                X_train_dir,
                y_train_dir,
                param_dist=tune_param_dist,
                n_iter=tune_n_iter,
                cv_splits=tune_cv_splits,
                scoring=tune_scoring,
                scale=True,
                sample_weight=w_train_dir,
                verbose_per_fold=tune_verbose_per_fold,
                show_accuracy=tune_show_accuracy,
            )
            print("[TRAIN] DIR best params:", best_params)
        else:
            model_dir_stage.fit(
                X_train_dir,
                y_train_dir,
                X_val=X_val_dir,
                y_val=y_val_dir,
                feature_names=feature_cols,
                sample_weight=w_train_dir,
                sample_weight_val=w_val_dir,
            )

        move_pred = model_move.predict(X_test)
        dir_pred_full = model_dir_stage.predict(X_test)
        y_pred = np.where(move_pred == 1, np.where(dir_pred_full == 1, 2, 0), 1)

        artifact = {
            "mode": "two_stage",
            "move": {"model": model_move.model, "scaler": model_move.scaler},
            "dir": {"model": model_dir_stage.model, "scaler": model_dir_stage.scaler},
            "feature_names": feature_cols,
            "config": params,
        }
    else:
        print("[TRAIN] Mode:", "BINARY" if is_binary else "3-CLASS")
        model = XGBBaseline(
            n_classes=n_classes,
            device=device,
            random_state=42,
            **xgb_params,
        )
        w_train = compute_sample_weight(class_weight="balanced", y=y_train)
        w_val = compute_sample_weight(class_weight="balanced", y=y_val)
        if tune_enabled:
            print("[TRAIN] Tuning enabled:",
                  f"n_iter={tune_n_iter}, cv_splits={tune_cv_splits}, scoring={tune_scoring}")
            print("[TRAIN] Tuning model...")
            best_params = model.tune(
                X_train,
                y_train,
                param_dist=tune_param_dist,
                n_iter=tune_n_iter,
                cv_splits=tune_cv_splits,
                scoring=tune_scoring,
                scale=True,
                sample_weight=w_train,
                verbose_per_fold=tune_verbose_per_fold,
                show_accuracy=tune_show_accuracy,
            )
            print("[TRAIN] Best params:", best_params)
        else:
            model.fit(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=feature_cols,
                sample_weight=w_train,
                sample_weight_val=w_val,
            )

        if is_binary and threshold_tuning:
            X_val_scaled = model.scaler.transform(X_val)
            proba_val = model.model.predict_proba(X_val_scaled)[:, 1]
            best_t, _ = _threshold_tune(proba_val, y_val, metric="bal_acc")

            X_test_scaled = model.scaler.transform(X_test)
            proba_test = model.model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (proba_test >= best_t).astype(np.int32)
        else:
            y_pred = model.predict(X_test)

        artifact = {
            "mode": "single_stage",
            "model": model.model,
            "scaler": model.scaler,
            "feature_names": feature_cols,
            "config": params,
        }

    # Save artifact
    dump(artifact, out_path)

    # Save metrics report
    label_names = ["DOWN", "UP"] if is_binary else ["DOWN", "SIDEWAYS", "UP"]
    metrics = calc_metrics(y_test, y_pred, labels=labels)
    report = classification_report_dict(y_test, y_pred, labels=labels, names=label_names)
    out_report = ctx.reports_dir / f"xgb_metrics_{tag}.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps({"metrics": metrics, "report": report}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    meta = build_meta(step="train", params=params, inputs=inputs)
    save_meta(meta_path_for(out_path), meta)
    print("[TRAIN] Saved model:", out_path)
    print("[TRAIN] Metrics report:", out_report)
    print("[TRAIN] Metrics:", metrics)
    # Per-class metrics
    try:
        print("[TRAIN] Per-class metrics:")
        counts = {int(k): int((y_test == k).sum()) for k in labels}
        total = len(y_test)
        for idx, name in zip(labels, label_names):
            key = name
            if key in report:
                r = report[key]
                freq = counts.get(int(idx), 0) / total * 100 if total else 0.0
                print(f"  {name}: acc={r.get('recall', 0):.4f}, f1={r.get('f1-score', 0):.4f}, "
                      f"freq={freq:.2f}%")
    except Exception as e:
        print(f"[TRAIN] Per-class metrics failed: {e}")
    return out_path
