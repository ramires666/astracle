"""
Evaluation plots step for astro_xgb pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext
from src.pipeline.astro_xgb.metrics import confusion_matrix_data


def _extract_tag(model_path: Path) -> str:
    return Path(model_path).stem.replace("xgb_astro_", "")


def run_eval_step(
    ctx: PipelineContext,
    model_path: Path,
    test_path: Path,
    force: bool = False,
    feature_importance_stage: str = "dir",
) -> Path:
    """
    Generate confusion matrix and feature importance plots.
    Returns confusion matrix image path.
    """
    tag = _extract_tag(model_path)
    out_cm = ctx.reports_dir / f"confusion_matrix_{tag}.png"
    out_fi = ctx.reports_dir / f"feature_importance_{tag}.png"

    params = {"feature_importance_stage": feature_importance_stage}
    inputs = [Path(model_path), Path(test_path)]

    if not force and is_cache_valid(out_cm, params=params, inputs=inputs, step="eval"):
        print("[EVAL] Using cached plots:")
        print("  confusion_matrix:", out_cm)
        print("  feature_importance:", out_fi)
        return out_cm

    artifact = load(model_path)
    df_test = pd.read_parquet(test_path)
    feature_cols = artifact.get("feature_names") or [c for c in df_test.columns if c not in ["date", "target"]]
    X_test = df_test[feature_cols].to_numpy(dtype=np.float64)
    y_test = df_test["target"].to_numpy(dtype=np.int32)

    if artifact.get("mode") == "two_stage":
        model_move = artifact["move"]["model"]
        scaler_move = artifact["move"]["scaler"]
        model_dir = artifact["dir"]["model"]
        scaler_dir = artifact["dir"]["scaler"]

        X_scaled_move = scaler_move.transform(X_test)
        move_pred = model_move.predict(X_scaled_move)
        X_scaled_dir = scaler_dir.transform(X_test)
        dir_pred = model_dir.predict(X_scaled_dir)
        y_pred = np.where(move_pred == 1, np.where(dir_pred == 1, 2, 0), 1)

        fi_model = model_dir if feature_importance_stage == "dir" else model_move
    else:
        model = artifact["model"]
        scaler = artifact["scaler"]
        X_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_scaled)
        fi_model = model

    labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    if labels == [0, 1]:
        label_names = ["DOWN", "UP"]
    elif labels == [0, 1, 2]:
        label_names = ["DOWN", "SIDEWAYS", "UP"]
    else:
        name_map = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}
        label_names = [name_map.get(int(lbl), str(lbl)) for lbl in labels]

    cm = confusion_matrix_data(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_cm, dpi=150)
    plt.close()

    # Feature importance
    try:
        importances = fi_model.feature_importances_
        order = np.argsort(importances)[::-1][:20]
        top_features = [feature_cols[i] for i in order]
        top_values = importances[order]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_values, y=top_features, color="tab:blue")
        plt.title("Top-20 Astro Features by Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(out_fi, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Feature importance plot failed: {e}")

    meta = build_meta(step="eval", params=params, inputs=inputs)
    save_meta(meta_path_for(out_cm), meta)
    print("[EVAL] Saved plots:")
    print("  confusion_matrix:", out_cm)
    print("  feature_importance:", out_fi)
    return out_cm
