"""
XGBoost baseline model for direction classification.

Used only for MVP (simple version).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import RobustScaler


# Avoid OpenMP conflicts on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class XGBBaseline:
    """
    Simple wrapper over XGBClassifier.
    """

    def __init__(
        self,
        n_classes: int = 3,
        device: str = "cpu",
        random_state: int = 42,
        **kwargs,
    ):
        self.n_classes = n_classes
        self.device = device
        self.random_state = random_state
        self.feature_names: List[str] = []

        default_eval_metric = "mlogloss" if n_classes > 2 else "logloss"
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": default_eval_metric,
            "device": device,
            "random_state": random_state,
        }
        if n_classes > 2 and "objective" not in kwargs:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
        params.update(kwargs)

        self.model = XGBClassifier(**params)
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.constant_class: Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the model.
        """
        self.constant_class = None

        if feature_names:
            self.feature_names = feature_names

        unique_train = np.unique(y_train)
        if unique_train.size < 2:
            # Fit scaler for consistent preprocessing, but skip model training.
            self.scaler.fit(X_train)
            self.constant_class = int(unique_train[0]) if unique_train.size == 1 else 0
            self.is_fitted = True
            warnings.warn(
                "XGBBaseline: only one class in training data; "
                "skipping model fit and using constant predictor.",
                RuntimeWarning,
            )
            return

        # Sample weights help with class imbalance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        if X_val_scaled is not None and y_val is not None:
            eval_set = [(X_val_scaled, y_val)]
            eval_weights = [sample_weight_val] if sample_weight_val is not None else None
            self.model.fit(
                X_train_scaled,
                y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                sample_weight_eval_set=eval_weights,
                verbose=False,
            )
        else:
            self.model.fit(
                X_train_scaled,
                y_train,
                sample_weight=sample_weight,
                verbose=False,
            )

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")

        if self.constant_class is not None:
            return np.full(X.shape[0], self.constant_class, dtype=np.int32)

        X_scaled = self.scaler.transform(X)

        # If model on GPU but data on CPU, avoid the warning
        # by temporarily switching device to CPU for predict.
        restore_device = None
        if str(self.device).startswith("cuda"):
            try:
                import cupy as cp  # type: ignore
                is_gpu_array = isinstance(X_scaled, cp.ndarray)
            except Exception:
                is_gpu_array = False

            if not is_gpu_array:
                try:
                    self.model.get_booster().set_param({"device": "cpu"})
                    restore_device = self.device
                except Exception:
                    restore_device = None

        try:
            preds = self.model.predict(X_scaled)
        finally:
            if restore_device is not None:
                try:
                    self.model.get_booster().set_param({"device": restore_device})
                except Exception:
                    pass

        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model quality.
        """
        y_pred = self.predict(X)
        # target_names must match the number of classes
        unique_labels = np.unique(np.concatenate([y, y_pred]))
        if self.n_classes == 3 and len(unique_labels) == 3:
            target_names = ["DOWN", "SIDEWAYS", "UP"]
        elif self.n_classes == 2 and len(unique_labels) == 2:
            target_names = ["DOWN", "UP"]
        else:
            # If class count mismatch, show simple labels
            target_names = [str(v) for v in unique_labels]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "classification_report": classification_report(
                y,
                y_pred,
                labels=unique_labels,
                target_names=target_names,
            ),
        }
        return metrics
