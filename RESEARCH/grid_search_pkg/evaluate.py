"""
Evaluate functions for Grid Search.

Contains evaluate_combo and related evaluation functions.
"""
import pandas as pd
from typing import Dict, List, Optional, Any

from ..labeling import create_balanced_labels
from ..astro_engine import (
    calculate_aspects_for_dates,
    calculate_aspects_from_cache,
    calculate_phases_for_dates,
)
from ..features import build_full_features, merge_features_with_labels
from ..model_training import (
    split_dataset,
    prepare_xy,
    train_xgb_model,
    tune_threshold,
    predict_with_threshold,
    calc_metrics,
)


def evaluate_combo(
    df_market: pd.DataFrame,
    df_bodies: pd.DataFrame,
    bodies_by_date: dict,
    settings: Any,
    orb_mult: float,
    gauss_window: int,
    gauss_std: float,
    exclude_bodies: Optional[List[str]] = None,
    angles_cache: Optional[dict] = None,
    device: str = "cpu",
    model_params: Optional[Dict] = None,
) -> Dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    ОЦЕНКА ОДНОЙ КОМБИНАЦИИ ГИПЕРПАРАМЕТРОВ
    ═══════════════════════════════════════════════════════════════════════════════
    
    Эта функция:
    1. Создаёт разметку (UP/DOWN) с заданными gauss_window и gauss_std
    2. Вычисляет аспекты с заданным orb_mult (или использует кэш!)
    3. Вычисляет фазы Луны и элонгации планет
    4. Строит признаки (исключая exclude_bodies!) и объединяет с разметкой
    5. Обучает XGBoost и возвращает метрики
    
    ОПТИМИЗАЦИЯ:
    ─────────────────────────────────────────────────────────────────────────────
    Если передан angles_cache, аспекты рассчитываются из кэша (~3-5x быстрее).
    Кэш создаётся один раз функцией precompute_angles_for_dates().
    ═══════════════════════════════════════════════════════════════════════════════
    
    Returns:
        Dictionary with combo params and metrics
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Создаём разметку (метки UP/DOWN) с заданными параметрами Гаусса
    # ─────────────────────────────────────────────────────────────────────────────
    df_labels = create_balanced_labels(
        df_market,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Вычисляем аспекты (используем кэш если есть!)
    # ─────────────────────────────────────────────────────────────────────────────
    if angles_cache is not None:
        # Быстрый путь: используем предвычисленные углы
        df_aspects = calculate_aspects_from_cache(
            angles_cache, settings, orb_mult=orb_mult, progress=False
        )
    else:
        # Медленный путь: пересчитываем углы
        df_aspects = calculate_aspects_for_dates(
            bodies_by_date, settings, orb_mult=orb_mult, progress=False
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 3: Вычисляем фазы Луны и элонгации планет
    # ─────────────────────────────────────────────────────────────────────────────
    df_phases = calculate_phases_for_dates(bodies_by_date, progress=False)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 4: Строим полную матрицу признаков (исключая указанные тела!)
    # ─────────────────────────────────────────────────────────────────────────────
    df_features = build_full_features(
        df_bodies, df_aspects, 
        df_phases=df_phases,
        exclude_bodies=exclude_bodies
    )
    
    # Merge with labels
    df_dataset = merge_features_with_labels(df_features, df_labels)
    
    if len(df_dataset) < 100:
        return {"error": "Too few samples"}
    
    # Split
    train_df, val_df, test_df = split_dataset(df_dataset)
    
    feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)
    
    # Train
    params = model_params or {}
    model = train_xgb_model(
        X_train, y_train, X_val, y_val,
        feature_cols, n_classes=2, device=device,
        **params
    )
    
    # Tune threshold по recall_min (а не bal_acc)
    best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min")
    
    # Predict on test
    y_pred = predict_with_threshold(model, X_test, threshold=best_t)
    
    # Metrics
    metrics = calc_metrics(y_test, y_pred, [0, 1])
    
    # Per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["DOWN", "UP"], output_dict=True, zero_division=0
    )
    
    f1_down = report["DOWN"]["f1-score"]
    f1_up = report["UP"]["f1-score"]
    recall_down = report["DOWN"]["recall"]
    recall_up = report["UP"]["recall"]
    
    return {
        "orb_mult": orb_mult,
        "gauss_window": gauss_window,
        "gauss_std": gauss_std,
        "exclude_bodies": exclude_bodies or [],
        "threshold": best_t,
        "recall_down": recall_down,
        "recall_up": recall_up,
        "recall_min": min(recall_down, recall_up),
        "recall_gap": abs(recall_down - recall_up),
        "f1_down": f1_down,
        "f1_up": f1_up,
        "f1_min": min(f1_down, f1_up),
        "f1_gap": abs(f1_down - f1_up),
        "f1_macro": metrics["f1_macro"],
        "bal_acc": metrics["bal_acc"],
        "mcc": metrics["mcc"],
        "summary": metrics["summary"],
    }
