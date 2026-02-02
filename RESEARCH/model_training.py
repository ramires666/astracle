"""
Model training module for RESEARCH pipeline.
XGBoost training, evaluation, and metrics.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
    matthews_corrcoef,
)

from src.models.xgb import XGBBaseline

from .features import get_feature_columns


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/val/test split (no shuffling).
    
    Args:
        df: Dataset with date, features, and target
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X, y arrays from DataFrame.
    
    Args:
        df: DataFrame with features and target
        feature_cols: Feature column names (auto-detected if None)
    
    Returns:
        Tuple of (X, y)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.int32)
    
    return X, y


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict:
    """Calculate standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "acc": acc,
        "bal_acc": bal,
        "mcc": mcc,
        "f1_macro": f1m,
        "summary": 0.5 * (bal + f1m),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    n_boot: int = 200,
    seed: int = 42,
) -> Dict:
    """Calculate bootstrap confidence intervals for metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    if n == 0:
        return None
    
    samples = {"acc": [], "bal_acc": [], "mcc": [], "f1_macro": [], "summary": []}
    
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = calc_metrics(y_true[idx], y_pred[idx], labels)
        for k in samples:
            samples[k].append(m[k])
    
    out = {}
    for k, vals in samples.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[k] = (float(lo), float(hi))
    
    return out


def majority_baseline_pred(y_true: np.ndarray, labels: List[int]) -> np.ndarray:
    """Baseline: always predict majority class."""
    counts = [int((y_true == lbl).sum()) for lbl in labels]
    majority = labels[int(np.argmax(counts))]
    return np.full_like(y_true, majority)


def prev_label_baseline_pred(y_true: np.ndarray, fallback: int = 0) -> np.ndarray:
    """Baseline: predict previous label (naive time baseline)."""
    if len(y_true) == 0:
        return np.array([], dtype=y_true.dtype)
    pred = np.roll(y_true, 1)
    pred[0] = fallback
    return pred


def train_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_classes: int = 2,
    device: str = "cpu",
    **model_params,
) -> XGBBaseline:
    """
    Train XGBoost model with balanced sample weights.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: Feature column names
        n_classes: Number of classes
        device: 'cpu' or 'cuda'
        **model_params: Additional XGBoost parameters
    
    Returns:
        Trained XGBBaseline model
    """
    # Compute sample weights
    w_train = compute_sample_weight(class_weight="balanced", y=y_train)
    w_val = compute_sample_weight(class_weight="balanced", y=y_val)
    
    # Default params
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    }
    params.update(model_params)
    
    model = XGBBaseline(
        n_classes=n_classes,
        device=device,
        random_state=42,
        **params,
    )
    
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        feature_names=feature_names,
        sample_weight=w_train,
        sample_weight_val=w_val,
    )
    
    return model


def tune_threshold(
    model: XGBBaseline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "bal_acc",
) -> Tuple[float, float]:
    """
    Tune probability threshold on validation set.
    
    Args:
        model: Trained XGBBaseline model
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('bal_acc', 'f1_macro', 'summary')
    
    Returns:
        Tuple of (best_threshold, best_score)
    """
    X_scaled = model.scaler.transform(X_val)
    proba = model.model.predict_proba(X_scaled)[:, 1]
    
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_score = -1.0
    
    for t in thresholds:
        pred = (proba >= t).astype(np.int32)
        m = calc_metrics(y_val, pred, [0, 1])
        score = m.get(metric, m["bal_acc"])
        if score > best_score:
            best_score = score
            best_t = float(t)
    
    print(f"Best threshold={best_t:.2f}, {metric}={best_score:.4f}")
    return best_t, best_score


def predict_with_threshold(
    model: XGBBaseline,
    X: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict using custom probability threshold."""
    X_scaled = model.scaler.transform(X)
    proba = model.model.predict_proba(X_scaled)[:, 1]
    return (proba >= threshold).astype(np.int32)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ["DOWN", "UP"],
    print_report: bool = True,
) -> Dict:
    """
    Comprehensive model evaluation with baselines.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Class names
        print_report: Print detailed report
    
    Returns:
        Dictionary with metrics
    """
    labels = list(range(len(label_names)))
    
    metrics = calc_metrics(y_true, y_pred, labels)
    
    # Baselines
    base_pred = majority_baseline_pred(y_true, labels)
    base_metrics = calc_metrics(y_true, base_pred, labels)
    
    prev_pred = prev_label_baseline_pred(y_true, fallback=labels[0])
    prev_metrics = calc_metrics(y_true, prev_pred, labels)
    
    if print_report:
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Balanced Accuracy: {metrics['bal_acc']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"Summary Score: {metrics['summary']:.4f}")
        
        print(f"\nMajority Baseline: acc={base_metrics['acc']:.4f}, bal_acc={base_metrics['bal_acc']:.4f}")
        print(f"Prev-Label Baseline: acc={prev_metrics['acc']:.4f}, bal_acc={prev_metrics['bal_acc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels, target_names=label_names, zero_division=0))
        
        # Bootstrap CI
        ci = bootstrap_metrics(y_true, y_pred, labels)
        if ci:
            print("95% Bootstrap CI:")
            for k in ["acc", "bal_acc", "f1_macro"]:
                lo, hi = ci[k]
                print(f"  {k}: [{lo:.4f}, {hi:.4f}]")
    
    return {
        "metrics": metrics,
        "baseline_majority": base_metrics,
        "baseline_prev": prev_metrics,
    }


def get_feature_importance(
    model: XGBBaseline,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Get feature importance ranking."""
    importances = model.model.feature_importances_
    
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    return imp_df.head(top_n)


def check_cuda_available() -> Tuple[bool, str]:
    """Check if CUDA is available for XGBoost."""
    try:
        import xgboost as xgb
        info = xgb.build_info()
        use_cuda = bool(info.get("USE_CUDA", False))
        return use_cuda, "cuda" if use_cuda else "cpu"
    except:
        return False, "cpu"
