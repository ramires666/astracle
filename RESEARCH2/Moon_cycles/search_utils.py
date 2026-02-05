"""Training and Gaussian-search helpers for Moon-cycle research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from RESEARCH.cache_utils import load_cache, save_cache
from RESEARCH.model_training import (
    check_cuda_available,
    prepare_xy,
    predict_with_threshold,
    train_xgb_model,
    tune_threshold,
)

from .eval_utils import (
    compute_binary_metrics,
    compute_statistical_significance,
    make_coin_flip_baseline,
    make_majority_baseline,
)
from .moon_data import MoonLabelConfig, build_moon_dataset_for_gauss, get_moon_feature_columns
from .progress_utils import progress_update
from .threshold_utils import predict_proba_up_safe, tune_threshold_with_balance
from .splits import (
    SplitDefinition,
    describe_split,
    make_classic_split,
    make_walk_forward_splits,
)


@dataclass(frozen=True)
class XgbConfig:
    """Model hyperparameters used by all experiments."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.03
    colsample_bytree: float = 0.8
    subsample: float = 0.8
    early_stopping_rounds: int = 50
    threshold_gap_penalty: float = 0.25
    threshold_prior_penalty: float = 0.05


@dataclass(frozen=True)
class WalkForwardConfig:
    """Split protocol settings for walk-forward evaluation."""

    warmup_ratio: float = 0.50
    block_ratios: tuple[float, ...] = (0.10, 0.10, 0.10, 0.10, 0.10)
    val_fraction_inside_block: float = 0.50


def _empty_predictions_frame(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a standard prediction frame that we fill during evaluation."""
    out = df_dataset[["date", "close", "target"]].copy().reset_index(drop=True)
    out["split_role"] = "train"
    out["pred_label"] = np.nan
    out["pred_proba_up"] = np.nan
    out["baseline_majority"] = np.nan
    out["baseline_random"] = np.nan
    return out


def _run_one_split(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    split: SplitDefinition,
    model_cfg: XgbConfig,
    device: str,
    threshold_metric: str = "recall_min",
) -> Dict[str, object]:
    """Train and evaluate one split (used by classic and walk-forward protocols)."""
    train_df = df_dataset.iloc[split.train_idx].copy().reset_index(drop=True)
    val_df = df_dataset.iloc[split.val_idx].copy().reset_index(drop=True)
    test_df = df_dataset.iloc[split.test_idx].copy().reset_index(drop=True)

    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    # "Before training" baselines.
    base_majority_test = make_majority_baseline(y_train=y_train, size=len(y_test))
    train_up_share = float((y_train == 1).mean()) if len(y_train) > 0 else 0.5
    base_random_test = make_coin_flip_baseline(size=len(y_test), p_up=train_up_share, seed=42)

    metrics_base_majority = compute_binary_metrics(y_true=y_test, y_pred=base_majority_test)
    metrics_base_random = compute_binary_metrics(y_true=y_test, y_pred=base_random_test)

    # Train + tune threshold on validation.
    model = train_xgb_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_cols,
        n_classes=2,
        device=device,
        verbose=False,
        early_stopping_rounds=model_cfg.early_stopping_rounds,
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        colsample_bytree=model_cfg.colsample_bytree,
        subsample=model_cfg.subsample,
    )

    # Balance-aware threshold tuning by default; explicit metric uses legacy tuner.
    if str(threshold_metric).lower() in {"balanced", "recall_min"}:
        proba_val = predict_proba_up_safe(model=model, X=X_val)
        best_threshold, val_target_score = tune_threshold_with_balance(
            y_val=y_val,
            proba_up=proba_val,
            gap_penalty=model_cfg.threshold_gap_penalty,
            prior_penalty=model_cfg.threshold_prior_penalty,
        )
    else:
        best_threshold, val_target_score = tune_threshold(
            model=model,
            X_val=X_val,
            y_val=y_val,
            metric=threshold_metric,
            verbose=False,
        )

    y_pred_train = predict_with_threshold(model=model, X=X_train, threshold=best_threshold)
    y_pred_val = predict_with_threshold(model=model, X=X_val, threshold=best_threshold)
    y_pred_test = predict_with_threshold(model=model, X=X_test, threshold=best_threshold)

    proba_test = predict_proba_up_safe(model=model, X=X_test)

    metrics_train = compute_binary_metrics(y_true=y_train, y_pred=y_pred_train)
    metrics_val = compute_binary_metrics(y_true=y_val, y_pred=y_pred_val)
    metrics_test = compute_binary_metrics(y_true=y_test, y_pred=y_pred_test)
    significance_test = compute_statistical_significance(y_true=y_test, y_pred=y_pred_test)

    return {
        "split_meta": describe_split(df_dataset, split),
        "threshold": float(best_threshold),
        "val_target_score": float(val_target_score),
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "significance_test": significance_test,
        "baseline_majority_test": metrics_base_majority,
        "baseline_random_test": metrics_base_random,
        "class_share_train_up": float((y_train == 1).mean()) if len(y_train) > 0 else 0.5,
        "class_share_val_up": float((y_val == 1).mean()) if len(y_val) > 0 else 0.5,
        "class_share_test_up": float((y_test == 1).mean()) if len(y_test) > 0 else 0.5,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "proba_test": proba_test,
        "test_index": split.test_idx,
        "val_index": split.val_idx,
        "base_majority_test_pred": base_majority_test,
        "base_random_test_pred": base_random_test,
    }


def run_classic_protocol(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    model_cfg: XgbConfig,
    device: str,
    threshold_metric: str = "recall_min",
) -> Dict[str, object]:
    """Run one classic 70/15/15 experiment and return detailed artifacts."""
    split = make_classic_split(df_dataset)
    split_result = _run_one_split(
        df_dataset=df_dataset,
        feature_cols=feature_cols,
        split=split,
        model_cfg=model_cfg,
        device=device,
        threshold_metric=threshold_metric,
    )

    # Build row-level prediction table for charting.
    pred_df = _empty_predictions_frame(df_dataset)
    pred_df.loc[split.train_idx, "split_role"] = "train"
    pred_df.loc[split.val_idx, "split_role"] = "val"
    pred_df.loc[split.test_idx, "split_role"] = "test"

    pred_df.loc[split.test_idx, "pred_label"] = split_result["y_pred_test"]
    pred_df.loc[split.test_idx, "pred_proba_up"] = split_result["proba_test"]
    pred_df.loc[split.test_idx, "baseline_majority"] = split_result["base_majority_test_pred"]
    pred_df.loc[split.test_idx, "baseline_random"] = split_result["base_random_test_pred"]

    summary = {
        "protocol": "classic_70_15_15",
        "train_rows": int(len(split.train_idx)),
        "val_rows": int(len(split.val_idx)),
        "test_rows": int(len(split.test_idx)),
        "threshold": float(split_result["threshold"]),
        "val_target_score": float(split_result["val_target_score"]),
        "train_acc": float(split_result["metrics_train"]["accuracy"]),
        "val_acc": float(split_result["metrics_val"]["accuracy"]),
        "test_acc": float(split_result["metrics_test"]["accuracy"]),
        "test_bal_acc": float(split_result["metrics_test"]["balanced_accuracy"]),
        "test_mcc": float(split_result["metrics_test"]["mcc"]),
        "test_recall_min": float(split_result["metrics_test"]["recall_min"]),
        "test_recall_down": float(split_result["metrics_test"]["recall_down"]),
        "test_recall_up": float(split_result["metrics_test"]["recall_up"]),
        "test_recall_gap": float(split_result["metrics_test"]["recall_gap"]),
        "train_up_share": float(split_result["class_share_train_up"]),
        "val_up_share": float(split_result["class_share_val_up"]),
        "test_up_share": float(split_result["class_share_test_up"]),
        "train_target_imbalance": float(abs(0.5 - split_result["class_share_train_up"])),
        "val_target_imbalance": float(abs(0.5 - split_result["class_share_val_up"])),
        "test_target_imbalance": float(abs(0.5 - split_result["class_share_test_up"])),
        "baseline_majority_test_acc": float(split_result["baseline_majority_test"]["accuracy"]),
        "baseline_random_test_acc": float(split_result["baseline_random_test"]["accuracy"]),
        "p_value_vs_random": float(split_result["significance_test"]["p_value_vs_random"]),
    }

    return {
        "summary": summary,
        "split_result": split_result,
        "predictions": pred_df,
    }


def run_walk_forward_protocol(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    model_cfg: XgbConfig,
    wf_cfg: WalkForwardConfig,
    device: str,
    threshold_metric: str = "recall_min",
) -> Dict[str, object]:
    """Run walk-forward protocol and aggregate all fold results."""
    splits = make_walk_forward_splits(
        df=df_dataset,
        warmup_ratio=wf_cfg.warmup_ratio,
        block_ratios=wf_cfg.block_ratios,
        val_fraction_inside_block=wf_cfg.val_fraction_inside_block,
    )

    fold_results: List[Dict[str, object]] = []
    pred_df = _empty_predictions_frame(df_dataset)

    for split in splits:
        fold = _run_one_split(
            df_dataset=df_dataset,
            feature_cols=feature_cols,
            split=split,
            model_cfg=model_cfg,
            device=device,
            threshold_metric=threshold_metric,
        )
        fold_results.append(fold)

        # We fill only current fold val/test region. Train regions are fold-specific
        # and overlap heavily, so they are less useful for one merged chart.
        pred_df.loc[split.val_idx, "split_role"] = "val"
        pred_df.loc[split.test_idx, "split_role"] = "test"
        pred_df.loc[split.test_idx, "pred_label"] = fold["y_pred_test"]
        pred_df.loc[split.test_idx, "pred_proba_up"] = fold["proba_test"]
        pred_df.loc[split.test_idx, "baseline_majority"] = fold["base_majority_test_pred"]
        pred_df.loc[split.test_idx, "baseline_random"] = fold["base_random_test_pred"]

    # Aggregate using concatenated test predictions across all folds.
    test_mask = pred_df["split_role"] == "test"
    y_true_all = pred_df.loc[test_mask, "target"].to_numpy(dtype=np.int32)
    y_pred_all = pred_df.loc[test_mask, "pred_label"].to_numpy(dtype=np.int32)
    y_base_majority_all = pred_df.loc[test_mask, "baseline_majority"].to_numpy(dtype=np.int32)
    y_base_random_all = pred_df.loc[test_mask, "baseline_random"].to_numpy(dtype=np.int32)

    agg_metrics_test = compute_binary_metrics(y_true=y_true_all, y_pred=y_pred_all)
    agg_base_majority = compute_binary_metrics(y_true=y_true_all, y_pred=y_base_majority_all)
    agg_base_random = compute_binary_metrics(y_true=y_true_all, y_pred=y_base_random_all)
    agg_significance = compute_statistical_significance(y_true=y_true_all, y_pred=y_pred_all)

    fold_table_rows = []
    for i, fold in enumerate(fold_results, start=1):
        fold_table_rows.append(
            {
                "fold": i,
                "threshold": float(fold["threshold"]),
                "val_target_score": float(fold["val_target_score"]),
                "test_acc": float(fold["metrics_test"]["accuracy"]),
                "test_mcc": float(fold["metrics_test"]["mcc"]),
                "test_recall_min": float(fold["metrics_test"]["recall_min"]),
                "test_recall_down": float(fold["metrics_test"]["recall_down"]),
                "test_recall_up": float(fold["metrics_test"]["recall_up"]),
                "test_recall_gap": float(fold["metrics_test"]["recall_gap"]),
            }
        )
    fold_table = pd.DataFrame(fold_table_rows)

    summary = {
        "protocol": "walk_forward",
        "n_folds": int(len(fold_results)),
        "test_rows_total": int(test_mask.sum()),
        "threshold_mean": float(np.mean([f["threshold"] for f in fold_results])),
        "test_acc": float(agg_metrics_test["accuracy"]),
        "test_bal_acc": float(agg_metrics_test["balanced_accuracy"]),
        "test_mcc": float(agg_metrics_test["mcc"]),
        "test_recall_min": float(agg_metrics_test["recall_min"]),
        "test_recall_down": float(agg_metrics_test["recall_down"]),
        "test_recall_up": float(agg_metrics_test["recall_up"]),
        "test_recall_gap": float(agg_metrics_test["recall_gap"]),
        "test_up_share": float((y_true_all == 1).mean()) if len(y_true_all) > 0 else 0.5,
        "test_target_imbalance": float(abs(0.5 - (y_true_all == 1).mean())) if len(y_true_all) > 0 else 0.0,
        "pred_up_share": float((y_pred_all == 1).mean()) if len(y_pred_all) > 0 else 0.5,
        "pred_target_gap": float(abs((y_pred_all == 1).mean() - (y_true_all == 1).mean())) if len(y_true_all) > 0 else 0.0,
        "baseline_majority_test_acc": float(agg_base_majority["accuracy"]),
        "baseline_random_test_acc": float(agg_base_random["accuracy"]),
        "p_value_vs_random": float(agg_significance["p_value_vs_random"]),
    }

    return {
        "summary": summary,
        "fold_table": fold_table,
        "fold_results": fold_results,
        "predictions": pred_df,
    }


def _search_cache_key(
    protocol: str,
    gauss_window: int,
    gauss_std: float,
    label_cfg: MoonLabelConfig,
    model_cfg: XgbConfig,
    wf_cfg: WalkForwardConfig,
) -> Dict[str, object]:
    """Build stable cache key for one experiment configuration."""
    return {
        "protocol": protocol,
        "gauss_window": int(gauss_window),
        "gauss_std": float(gauss_std),
        "label_cfg": asdict(label_cfg),
        "model_cfg": asdict(model_cfg),
        "wf_cfg": asdict(wf_cfg),
        "schema": "moon_search_v2",
    }


def run_gauss_search(
    df_market: pd.DataFrame,
    df_moon_features: pd.DataFrame,
    gauss_windows: Sequence[int],
    gauss_stds: Sequence[float],
    label_cfg: MoonLabelConfig,
    model_cfg: XgbConfig = XgbConfig(),
    wf_cfg: WalkForwardConfig = WalkForwardConfig(),
    protocol: str = "classic",
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run Gaussian parameter search for selected protocol.

    Protocol values:
    - "classic": one 70/15/15 split
    - "walk_forward": expanding-window folds
    """
    protocol = str(protocol).strip().lower()
    if protocol not in {"classic", "walk_forward"}:
        raise ValueError("protocol must be 'classic' or 'walk_forward'.")

    _, device = check_cuda_available()
    combos = list(product(gauss_windows, gauss_stds))

    rows: List[Dict[str, object]] = []
    detailed_runs: List[Dict[str, object]] = []

    total_runs, done_runs, best_so_far = int(len(combos)), 0, None  # Progress counters + best-so-far snapshot.
    for gauss_window, gauss_std in combos:
        key = _search_cache_key(
            protocol=protocol,
            gauss_window=gauss_window,
            gauss_std=gauss_std,
            label_cfg=label_cfg,
            model_cfg=model_cfg,
            wf_cfg=wf_cfg,
        )

        cached = load_cache("research2_moon", "search_run", key, verbose=verbose) if use_cache else None
        source = "cached" if cached is not None else "computed"
        if cached is not None:
            run_result = cached
        else:
            df_dataset = build_moon_dataset_for_gauss(
                df_market=df_market,
                df_moon_features=df_moon_features,
                gauss_window=int(gauss_window),
                gauss_std=float(gauss_std),
                label_cfg=label_cfg,
                use_cache=use_cache,
                verbose=verbose,
            )
            feature_cols = get_moon_feature_columns(df_dataset)

            if protocol == "classic":
                run_result = run_classic_protocol(
                    df_dataset=df_dataset,
                    feature_cols=feature_cols,
                    model_cfg=model_cfg,
                    device=device,
                )
            else:
                run_result = run_walk_forward_protocol(
                    df_dataset=df_dataset,
                    feature_cols=feature_cols,
                    model_cfg=model_cfg,
                    wf_cfg=wf_cfg,
                    device=device,
                )

            if use_cache:
                save_cache(run_result, "research2_moon", "search_run", key, verbose=verbose)

        row = dict(run_result["summary"])
        row["gauss_window"] = int(gauss_window)
        row["gauss_std"] = float(gauss_std)

        # Backward-compatibility for old cached runs:
        # previous schema did not store `test_recall_gap`.
        # If missing, we reconstruct it from per-class recalls.
        if "test_recall_gap" not in row:
            recall_down = row.get("test_recall_down")
            recall_up = row.get("test_recall_up")
            if recall_down is not None and recall_up is not None:
                row["test_recall_gap"] = abs(float(recall_down) - float(recall_up))
            else:
                row["test_recall_gap"] = np.nan

        rows.append(row)
        detailed_runs.append(run_result)

        done_runs += 1
        best_so_far = progress_update(best_so_far, row, done_runs, total_runs, prefix="gauss_search", verbose=verbose, source=source)

    # Ranking priority: maximize recall_min, minimize recall_gap, maximize MCC, then accuracy.
    df_results = pd.DataFrame(rows).sort_values(
        ["test_recall_min", "test_recall_gap", "test_mcc", "test_acc"],
        ascending=[False, True, False, False],
    )

    if df_results.empty:
        raise RuntimeError("Search produced no results.")

    best_row = df_results.iloc[0]
    best_window = int(best_row["gauss_window"])
    best_std = float(best_row["gauss_std"])

    best_result = None
    for run, row in zip(detailed_runs, rows):
        if int(row["gauss_window"]) == best_window and float(row["gauss_std"]) == best_std:
            best_result = run
            break

    if best_result is None:
        raise RuntimeError("Failed to locate detailed result for best Gaussian config.")

    return {
        "protocol": protocol,
        "results_table": df_results.reset_index(drop=True),
        "best_row": best_row.to_dict(),
        "best_result": best_result,
    }


def evaluate_fixed_gauss(
    df_market: pd.DataFrame,
    df_moon_features: pd.DataFrame,
    gauss_window: int,
    gauss_std: float,
    label_cfg: MoonLabelConfig,
    protocol: str,
    model_cfg: XgbConfig = XgbConfig(),
    wf_cfg: WalkForwardConfig = WalkForwardConfig(),
) -> Dict[str, object]:
    """
    Convenience helper: run one fixed Gaussian config without grid loop.

    Useful when we want to inspect one candidate in detail.
    """
    search = run_gauss_search(
        df_market=df_market,
        df_moon_features=df_moon_features,
        gauss_windows=[int(gauss_window)],
        gauss_stds=[float(gauss_std)],
        label_cfg=label_cfg,
        model_cfg=model_cfg,
        wf_cfg=wf_cfg,
        protocol=protocol,
        use_cache=True,
        verbose=True,
    )
    return search
