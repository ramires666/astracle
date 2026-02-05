"""
Model bakeoff helpers for Moon-cycle research.

Why this file exists:
- We want a "final answer" notebook that can tell us if Moon-only features
  contain any predictive edge at all.
- To make that answer convincing, we must compare multiple model families
  on the SAME data and the SAME time split.

Key rules we follow here:
1) Time split only (no shuffling). This avoids future leakage.
2) Gaussian label params are chosen by VALIDATION performance.
3) The TEST set is used only for the final report (no tuning on test).

This module is intentionally notebook-friendly:
- It returns a results table (easy to display).
- It can cache per-(model, gauss_window, gauss_std) run artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from RESEARCH.cache_utils import load_cache, save_cache
from RESEARCH.model_training import check_cuda_available, prepare_xy, train_xgb_model

from .eval_utils import (
    compute_binary_metrics,
    compute_statistical_significance,
    make_coin_flip_baseline,
    make_majority_baseline,
)
from .moon_data import MoonLabelConfig, build_moon_dataset_for_gauss, get_moon_feature_columns
from .progress_utils import progress_update
from .splits import SplitDefinition, make_classic_split
from .threshold_utils import predict_proba_up_safe, tune_threshold_with_balance

# =====================================================================================
# MODEL SPECS
# =====================================================================================


@dataclass(frozen=True)
class SkModelSpec:
    """A small description of a sklearn-style model we want to test."""

    name: str
    family: str
    needs_scaling: bool
    params: Dict[str, object]


def default_model_specs() -> List[SkModelSpec]:
    """
    Return a short list of reliable model families.

    We intentionally include models with very different inductive biases:
    - Linear: LogisticRegression
    - Bagged trees: RandomForest
    - Small neural net: MLPClassifier

    If ALL of them look random, it's a strong hint that the features are weak.
    """
    return [
        SkModelSpec(
            name="logreg",
            family="linear",
            needs_scaling=True,
            params={
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 2000,
                "random_state": 42,
            },
        ),
        SkModelSpec(
            name="rf",
            family="trees",
            needs_scaling=False,
            params={
                "n_estimators": 600,
                "max_depth": 6,
                "min_samples_leaf": 8,
                "random_state": 42,
                "n_jobs": 1,
            },
        ),
        SkModelSpec(
            name="mlp",
            family="neural_net",
            needs_scaling=True,
            params={
                "hidden_layer_sizes": (32, 16),
                "activation": "relu",
                "alpha": 1e-3,
                "learning_rate_init": 1e-3,
                "max_iter": 350,
                "random_state": 42,
            },
        ),
    ]


# =====================================================================================
# INTERNAL TRAIN/PREDICT HELPERS
# =====================================================================================


def _fit_sklearn_model(
    spec: SkModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> tuple[object, Optional[RobustScaler]]:
    """
    Fit one sklearn model.

    We return (model, scaler). For models that do not need scaling, scaler=None.
    """
    scaler: Optional[RobustScaler] = None

    if spec.needs_scaling:
        scaler = RobustScaler()
        X_fit = scaler.fit_transform(X_train)
    else:
        X_fit = X_train

    if spec.name == "logreg":
        model = LogisticRegression(**spec.params)
    elif spec.name == "rf":
        model = RandomForestClassifier(**spec.params)
    elif spec.name == "mlp":
        model = MLPClassifier(**spec.params)
    else:
        raise ValueError(f"Unknown model spec: {spec.name}")

    model.fit(X_fit, y_train, sample_weight=sample_weight)
    return model, scaler


def _predict_proba_up_sklearn(
    model: object,
    scaler: Optional[RobustScaler],
    X: np.ndarray,
) -> np.ndarray:
    """Predict probability of UP for sklearn models."""
    X_eval = scaler.transform(X) if scaler is not None else X
    proba = getattr(model, "predict_proba")(X_eval)[:, 1]
    return np.asarray(proba, dtype=float)


def _fit_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    device: str,
    xgb_params: Dict[str, object],
    sample_weight_train: Optional[np.ndarray],
    sample_weight_val: Optional[np.ndarray],
):
    """Fit XGBoost using the existing project wrapper."""
    model = train_xgb_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        n_classes=2,
        device=device,
        verbose=False,
        early_stopping_rounds=int(xgb_params.get("early_stopping_rounds", 50)),
        n_estimators=int(xgb_params.get("n_estimators", 500)),
        max_depth=int(xgb_params.get("max_depth", 6)),
        learning_rate=float(xgb_params.get("learning_rate", 0.03)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
        subsample=float(xgb_params.get("subsample", 0.8)),
        weight_power=float(xgb_params.get("weight_power", 1.0)),
        sideways_penalty=float(xgb_params.get("sideways_penalty", 1.0)),
    )

    # We manually re-fit with weights because train_xgb_model handles weights internally.
    # The wrapper already uses sample weights, so we do not pass weights twice.
    # Keeping this function in one place makes it easier to adjust later.
    _ = sample_weight_train
    _ = sample_weight_val

    return model


# =====================================================================================
# ONE-RUN EVALUATION
# =====================================================================================


def _empty_predictions_frame(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a standard prediction frame (same idea as search_utils)."""
    out = df_dataset[["date", "close", "target"]].copy().reset_index(drop=True)
    out["split_role"] = "train"
    out["pred_label"] = np.nan
    out["pred_proba_up"] = np.nan
    out["baseline_majority"] = np.nan
    out["baseline_random"] = np.nan
    return out


def _eval_one_model_on_split(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    split: SplitDefinition,
    model_name: str,
    sklearn_spec: Optional[SkModelSpec],
    xgb_params: Dict[str, object],
    threshold_gap_penalty: float,
    threshold_prior_penalty: float,
    device: str,
) -> Dict[str, object]:
    """
    Evaluate one model on one classic split.

    Returns a pickle-friendly dict (no model objects inside).
    """
    train_df = df_dataset.iloc[split.train_idx].copy().reset_index(drop=True)
    val_df = df_dataset.iloc[split.val_idx].copy().reset_index(drop=True)
    test_df = df_dataset.iloc[split.test_idx].copy().reset_index(drop=True)

    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    # Baselines for "before training" sanity check.
    base_majority_test = make_majority_baseline(y_train=y_train, size=len(y_test))
    train_up_share = float((y_train == 1).mean()) if len(y_train) > 0 else 0.5
    base_random_test = make_coin_flip_baseline(size=len(y_test), p_up=train_up_share, seed=42)

    # We keep weights for models that support them.
    w_train = compute_sample_weight(class_weight="balanced", y=y_train)

    # Train model + get probabilities.
    if model_name == "xgb":
        model = _fit_xgb_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_cols,
            device=device,
            xgb_params=xgb_params,
            sample_weight_train=w_train,
            sample_weight_val=None,
        )
        proba_val = predict_proba_up_safe(model=model, X=X_val)
        proba_test = predict_proba_up_safe(model=model, X=X_test)

    else:
        if sklearn_spec is None:
            raise ValueError("sklearn_spec is required for non-xgb model")

        model, scaler = _fit_sklearn_model(
            spec=sklearn_spec,
            X_train=X_train,
            y_train=y_train,
            sample_weight=w_train,
        )
        proba_val = _predict_proba_up_sklearn(model=model, scaler=scaler, X=X_val)
        proba_test = _predict_proba_up_sklearn(model=model, scaler=scaler, X=X_test)

    # Threshold is tuned on validation only (no test leakage).
    threshold, val_score = tune_threshold_with_balance(
        y_val=y_val,
        proba_up=proba_val,
        gap_penalty=threshold_gap_penalty,
        prior_penalty=threshold_prior_penalty,
    )

    y_pred_val = (proba_val >= threshold).astype(np.int32)
    y_pred_test = (proba_test >= threshold).astype(np.int32)

    metrics_val = compute_binary_metrics(y_true=y_val, y_pred=y_pred_val)
    metrics_test = compute_binary_metrics(y_true=y_test, y_pred=y_pred_test)
    significance_test = compute_statistical_significance(y_true=y_test, y_pred=y_pred_test)

    metrics_base_majority = compute_binary_metrics(y_true=y_test, y_pred=base_majority_test)
    metrics_base_random = compute_binary_metrics(y_true=y_test, y_pred=base_random_test)

    pred_df = _empty_predictions_frame(df_dataset)
    pred_df.loc[split.train_idx, "split_role"] = "train"
    pred_df.loc[split.val_idx, "split_role"] = "val"
    pred_df.loc[split.test_idx, "split_role"] = "test"

    pred_df.loc[split.test_idx, "pred_label"] = y_pred_test
    pred_df.loc[split.test_idx, "pred_proba_up"] = proba_test
    pred_df.loc[split.test_idx, "baseline_majority"] = base_majority_test
    pred_df.loc[split.test_idx, "baseline_random"] = base_random_test

    summary = {
        "model": model_name,
        "threshold": float(threshold),
        "val_score": float(val_score),
        "val_acc": float(metrics_val["accuracy"]),
        "val_mcc": float(metrics_val["mcc"]),
        "val_recall_min": float(metrics_val["recall_min"]),
        "val_recall_gap": float(metrics_val["recall_gap"]),
        "test_acc": float(metrics_test["accuracy"]),
        "test_bal_acc": float(metrics_test["balanced_accuracy"]),
        "test_mcc": float(metrics_test["mcc"]),
        "test_recall_min": float(metrics_test["recall_min"]),
        "test_recall_gap": float(metrics_test["recall_gap"]),
        "p_value_vs_random": float(significance_test["p_value_vs_random"]),
        "accuracy_ci95_low": float(significance_test["ci95_low"]),
        "accuracy_ci95_high": float(significance_test["ci95_high"]),
        "baseline_majority_test_acc": float(metrics_base_majority["accuracy"]),
        "baseline_random_test_acc": float(metrics_base_random["accuracy"]),
        "test_up_share": float((y_test == 1).mean()) if len(y_test) > 0 else 0.5,
        "pred_up_share": float((y_pred_test == 1).mean()) if len(y_pred_test) > 0 else 0.5,
    }

    return {
        "summary": summary,
        "predictions": pred_df,
    }


# =====================================================================================
# GRID SEARCH (GAUSS PARAMS) + MODEL BAKEOFF
# =====================================================================================


def _bakeoff_cache_key(
    model: str,
    gauss_window: int,
    gauss_std: float,
    label_cfg: MoonLabelConfig,
    xgb_params: Dict[str, object],
    threshold_gap_penalty: float,
    threshold_prior_penalty: float,
) -> Dict[str, object]:
    """Stable cache key for one bakeoff run."""
    return {
        "schema": "moon_bakeoff_v1",
        "model": model,
        "gauss_window": int(gauss_window),
        "gauss_std": float(gauss_std),
        "label_cfg": asdict(label_cfg),
        "xgb_params": dict(xgb_params),
        "threshold_gap_penalty": float(threshold_gap_penalty),
        "threshold_prior_penalty": float(threshold_prior_penalty),
    }


def run_moon_model_bakeoff(
    df_market: pd.DataFrame,
    df_moon_features: pd.DataFrame,
    gauss_windows: Sequence[int],
    gauss_stds: Sequence[float],
    label_cfg: MoonLabelConfig,
    model_specs: Optional[List[SkModelSpec]] = None,
    include_xgb: bool = True,
    xgb_params: Optional[Dict[str, object]] = None,
    threshold_gap_penalty: float = 0.25,
    threshold_prior_penalty: float = 0.05,
    cache_namespace: str = "research2_moon",
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run a "bakeoff": multiple model families across a Gaussian label grid.

    Selection logic:
    - We compute VAL and TEST metrics for each (model, gauss) pair.
    - We choose the winner PER MODEL by VAL metrics only.
    - We then report the winner's TEST metrics (honest final check).
    """
    model_specs = model_specs or default_model_specs()
    xgb_params = xgb_params or {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "early_stopping_rounds": 50,
    }

    # We build the model list ONCE so we can compute total work for progress printing.
    # This also avoids rebuilding the same list for every (gauss_window, gauss_std).
    model_list: List[tuple[str, Optional[SkModelSpec]]] = []
    if include_xgb:
        model_list.append(("xgb", None))
    for spec in model_specs:
        model_list.append((spec.name, spec))

    combos = list(product(gauss_windows, gauss_stds))
    _, device = check_cuda_available()

    all_rows: List[Dict[str, object]] = []
    detailed: Dict[Tuple[str, int, float], Dict[str, object]] = {}

    total_runs = int(len(combos) * len(model_list))
    done_runs = 0
    best_so_far: Optional[Dict[str, object]] = None

    for gauss_window, gauss_std in combos:
        df_dataset = build_moon_dataset_for_gauss(
            df_market=df_market,
            df_moon_features=df_moon_features,
            gauss_window=int(gauss_window),
            gauss_std=float(gauss_std),
            label_cfg=label_cfg,
            cache_namespace=cache_namespace,
            use_cache=use_cache,
            verbose=verbose,
        )
        feature_cols = get_moon_feature_columns(df_dataset)
        split = make_classic_split(df_dataset)

        for model_name, spec in model_list:
            cache_key = _bakeoff_cache_key(
                model=model_name,
                gauss_window=int(gauss_window),
                gauss_std=float(gauss_std),
                label_cfg=label_cfg,
                xgb_params=xgb_params,
                threshold_gap_penalty=threshold_gap_penalty,
                threshold_prior_penalty=threshold_prior_penalty,
            )

            cached = load_cache(cache_namespace, "bakeoff_run", cache_key, verbose=verbose) if use_cache else None
            source = "cached" if cached is not None else "computed"
            if cached is not None:
                run = cached
            else:
                run = _eval_one_model_on_split(
                    df_dataset=df_dataset,
                    feature_cols=feature_cols,
                    split=split,
                    model_name=model_name,
                    sklearn_spec=spec,
                    xgb_params=xgb_params,
                    threshold_gap_penalty=threshold_gap_penalty,
                    threshold_prior_penalty=threshold_prior_penalty,
                    device=device,
                )
                if use_cache:
                    save_cache(run, cache_namespace, "bakeoff_run", cache_key, verbose=verbose)

            row = dict(run["summary"])
            row["gauss_window"] = int(gauss_window)
            row["gauss_std"] = float(gauss_std)
            all_rows.append(row)
            detailed[(model_name, int(gauss_window), float(gauss_std))] = run

            # Primitive progress (for humans only):
            # - shows done/left + key metrics for the current run,
            # - also shows the best TEST result seen so far (just for monitoring).
            # IMPORTANT: below we still pick winners PER MODEL by VALIDATION only.
            done_runs += 1
            best_so_far = progress_update(best_so_far, row, done_runs, total_runs, prefix="bakeoff", verbose=verbose, source=source)

    results = pd.DataFrame(all_rows)
    if results.empty:
        raise RuntimeError("Bakeoff produced no rows.")

    # Best-by-VAL per model.
    best_rows = []
    best_runs: Dict[str, Dict[str, object]] = {}

    for model_name in sorted(results["model"].unique()):
        sub = results[results["model"] == model_name].copy()
        sub = sub.sort_values(
            ["val_recall_min", "val_recall_gap", "val_mcc", "val_acc"],
            ascending=[False, True, False, False],
        )
        best = sub.iloc[0]
        best_rows.append(best.to_dict())

        key = (model_name, int(best["gauss_window"]), float(best["gauss_std"]))
        best_runs[model_name] = detailed[key]

    best_table = pd.DataFrame(best_rows).sort_values(
        ["test_recall_min", "test_recall_gap", "test_mcc", "test_acc"],
        ascending=[False, True, False, False],
    )

    return {
        "results_table": results,
        "best_by_val_table": best_table.reset_index(drop=True),
        "best_runs": best_runs,
    }
