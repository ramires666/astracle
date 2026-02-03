"""
Grid search module for RESEARCH pipeline.
Hyperparameter optimization for orb multiplier, gaussian params, etc.

Saves best results to disk for later use.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# DataFrame display settings - no line wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

from .config import cfg
from .data_loader import load_market_data
from .labeling import create_balanced_labels, gaussian_smooth_centered
from .astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates,
    calculate_aspects_for_dates,
    calculate_transits_for_dates,
    get_natal_bodies,
    precompute_angles_for_dates,      # NEW: кэширование углов
    calculate_aspects_from_cache,     # NEW: аспекты из кэша
)
from .features import build_full_features, merge_features_with_labels
from .model_training import (
    split_dataset,
    prepare_xy,
    train_xgb_model,
    tune_threshold,
    predict_with_threshold,
    calc_metrics,
    check_cuda_available,
)


class GridSearchConfig:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    КОНФИГУРАЦИЯ GRID SEARCH (ПЕРЕБОР ПАРАМЕТРОВ)
    ═══════════════════════════════════════════════════════════════════════════════
    
    Grid Search — это метод поиска лучших параметров путём перебора всех комбинаций.
    Например, если у нас 3 значения orb и 3 значения gauss_std, мы проверим 3×3=9 комбинаций.
    
    ПАРАМЕТРЫ:
    ─────────────────────────────────────────────────────────────────────────────
    
    • orb_multipliers - множители орбиса аспектов
      [0.5] — узкие орбисы (только точные аспекты)
      [1.0] — стандартные орбисы
      [1.5] — широкие орбисы (больше аспектов)
      
    • gauss_windows - размер окна для сглаживания Гаусса (дней)
      [51]  — короткое окно (чувствительно к мелким движениям)
      [201] — длинное окно (ловит только большие тренды)
      
    • gauss_stds - стандартное отклонение Гаусса
      [30.0] — узкий колокол (чувствительная разметка)
      [70.0] — широкий колокол (плавная разметка)
      
    • coord_modes - системы координат для расчёта планет
      ["geo"]   — только геоцентрическая (Земля в центре, классика)
      ["helio"] — только гелиоцентрическая (Солнце в центре)
      ["both"]  — ОБЕ системы (удваивает количество признаков!)
      ["geo", "helio", "both"] — перебрать все три варианта
      
    • max_exclude - ABLATION: исключение астро-тел (NEW!)
      0 — не исключать тела (только orb/gauss/coord)
      1 — пробовать исключать по 1 телу
      2 — пробовать исключать до 2 тел
      4 — пробовать исключать до 4 тел (по умолчанию)
      
      Это КОМБИНАТОРНЫЙ взрыв! С 11 телами:
      - max_exclude=1: 11 вариантов
      - max_exclude=2: 66 вариантов
      - max_exclude=3: 231 вариант
      - max_exclude=4: 561 вариант
      
    • max_combos - ограничение количества комбинаций (для тестов)
    
    • model_params - параметры XGBoost модели
    ═══════════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        orb_multipliers: List[float] = [0.8, 1.0, 1.2],
        gauss_windows: List[int] = [101, 151, 201],
        gauss_stds: List[float] = [30.0, 50.0, 70.0],
        coord_modes: List[str] = ["geo"],  # geo, helio, both
        max_exclude: int = 0,  # NEW: 0 = без ablation, 4 = исключать до 4 тел
        max_combos: Optional[int] = None,
        model_params: Optional[Dict] = None,
    ):
        # ─────────────────────────────────────────────────────────────────────────
        # Сохраняем все параметры
        # ─────────────────────────────────────────────────────────────────────────
        self.orb_multipliers = orb_multipliers
        self.gauss_windows = gauss_windows
        self.gauss_stds = gauss_stds
        self.coord_modes = coord_modes
        self.max_exclude = max_exclude  # NEW: максимальное количество исключаемых тел
        self.max_combos = max_combos
        self.model_params = model_params or {
            "n_estimators": 500,    # Количество деревьев в ансамбле
            "max_depth": 3,         # Глубина каждого дерева (защита от переобучения)
            "learning_rate": 0.03,  # Скорость обучения (меньше = стабильнее)
            "subsample": 0.8,       # Доля данных для каждого дерева
            "colsample_bytree": 0.8,  # Доля признаков для каждого дерева
        }


def evaluate_combo(
    df_market: pd.DataFrame,
    df_bodies: pd.DataFrame,
    bodies_by_date: dict,
    settings: Any,
    orb_mult: float,
    gauss_window: int,
    gauss_std: float,
    exclude_bodies: Optional[List[str]] = None,
    angles_cache: Optional[dict] = None,  # NEW: предвычисленные углы
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
    from .astro_engine import calculate_phases_for_dates
    df_phases = calculate_phases_for_dates(bodies_by_date, progress=False)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 4: Строим полную матрицу признаков (исключая указанные тела!)
    # ─────────────────────────────────────────────────────────────────────────────
    df_features = build_full_features(
        df_bodies, df_aspects, 
        df_phases=df_phases,
        exclude_bodies=exclude_bodies  # NEW: исключаем указанные тела
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
        "exclude_bodies": exclude_bodies or [],  # NEW: какие тела исключены
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
        "mcc": metrics["mcc"],  # NEW: MCC метрика
        "summary": metrics["summary"],
    }


def run_grid_search(
    df_market: pd.DataFrame,
    config: Optional[GridSearchConfig] = None,
    save_results: bool = True,
    n_workers: int = 1,  # NEW: Number of parallel workers (1 = sequential)
) -> pd.DataFrame:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    GRID SEARCH - ПЕРЕБОР ВСЕХ КОМБИНАЦИЙ ПАРАМЕТРОВ
    ═══════════════════════════════════════════════════════════════════════════════
    
    Эта функция автоматически перебирает все комбинации параметров:
    • orb_multipliers - множители орбиса (широта аспектов)
    • gauss_windows - размер окна сглаживания
    • gauss_stds - стандартное отклонение Гаусса
    • coord_modes - системы координат (geo/helio/both)
    
    Для КАЖДОЙ комбинации:
    1. Рассчитываем позиции планет и аспекты
    2. Создаём разметку (UP/DOWN) с данными параметрами сглаживания
    3. Обучаем XGBoost модель
    4. Оцениваем качество на тестовых данных
    5. Сохраняем результат
    
    В конце сортируем по качеству (recall_min) и балансу (recall_gap).
    
    MULTIPROCESSING:
    ─────────────────────────────────────────────────────────────────────────────
    n_workers=1  — последовательное выполнение (по умолчанию)
    n_workers=4  — параллельно на 4 ядрах
    n_workers=-1 — все доступные ядра
    
    ⚠️ ВНИМАНИЕ: XGBoost сам использует многопоточность!
    Если у вас GPU, лучше n_workers=1 (GPU и так быстро).
    Если CPU, попробуйте n_workers=2-4.
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        df_market: Market data DataFrame
        config: GridSearchConfig (uses defaults if None)
        save_results: Save results to reports directory
        n_workers: Number of parallel workers (1=sequential, -1=all cores)
    
    Returns:
        DataFrame with all results sorted by balance
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Инициализация конфигурации
    # ─────────────────────────────────────────────────────────────────────────────
    config = config or GridSearchConfig()
    
    print("=" * 80)
    print("🔍 GRID SEARCH: ПЕРЕБОР ORB + GAUSSIAN + COORD_MODE")
    print("=" * 80)
    print(f"""
    Параметры поиска:
    • ORB множители:    {config.orb_multipliers}
    • GAUSS окна:       {config.gauss_windows}
    • GAUSS std:        {config.gauss_stds}
    • COORD режимы:     {config.coord_modes}
    """)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Проверяем CUDA (GPU) для ускорения XGBoost
    # ─────────────────────────────────────────────────────────────────────────────
    _, device = check_cuda_available()
    print(f"🖥️ Устройство: {device}")
    
    # Run timestamp for checkpoints
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 3: Инициализация Swiss Ephemeris
    # ─────────────────────────────────────────────────────────────────────────────
    settings = init_ephemeris()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 4: Генерируем ВСЕ комбинации параметров (включая ablation!)
    # ─────────────────────────────────────────────────────────────────────────────
    from itertools import combinations
    
    # Получаем список всех тел для ablation
    all_bodies = get_all_body_names(settings)
    
    # Генерируем комбинации исключений (если max_exclude > 0)
    exclusion_combos = [[]]  # Начинаем с пустого списка (baseline)
    
    if config.max_exclude > 0:
        for n_exclude in range(1, config.max_exclude + 1):
            for combo in combinations(all_bodies, n_exclude):
                exclusion_combos.append(list(combo))
    
    # Генерируем ВСЕ комбинации: (coord_mode, orb, gw, gs, exclude_bodies)
    combos = []
    for coord_mode in config.coord_modes:
        for orb in config.orb_multipliers:
            for gw in config.gauss_windows:
                for gs in config.gauss_stds:
                    for excl in exclusion_combos:
                        combos.append((coord_mode, orb, gw, gs, excl))
    
    # Ограничиваем количество комбинаций если задано
    if config.max_combos and len(combos) > config.max_combos:
        combos = combos[:config.max_combos]
    
    print(f"\n📊 Всего комбинаций для перебора: {len(combos)}")
    if config.max_exclude > 0:
        print(f"   (включая {len(exclusion_combos)} вариантов ablation: до {config.max_exclude} исключаемых тел)")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 5: Предварительно рассчитываем позиции планет для ВСЕХ режимов
    # ─────────────────────────────────────────────────────────────────────────────
    # Это экономит время - позиции планет одинаковы для всех orb/gauss комбо
    print("\n📍 Предварительный расчёт позиций планет...")
    
    from .astro_engine import calculate_bodies_for_dates_multi
    
    cached_bodies = {}  # Кэш: coord_mode -> (df_bodies, geo_by_date, helio_by_date)
    
    for coord_mode in config.coord_modes:
        if coord_mode not in cached_bodies:
            print(f"\n  → Режим {coord_mode.upper()}:")
            df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
                df_market["date"], settings, coord_mode=coord_mode, progress=True
            )
            cached_bodies[coord_mode] = (df_bodies, geo_by_date, helio_by_date)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 5.5: Предварительно рассчитываем УГЛЫ между планетами (ОПТИМИЗАЦИЯ!)
    # ─────────────────────────────────────────────────────────────────────────────
    # Углы не зависят от orb_mult — считаем один раз, фильтруем многократно
    print("\n📐 Предварительный расчёт углов между планетами...")
    
    cached_angles = {}  # Кэш: coord_mode -> angles_cache
    
    for coord_mode in config.coord_modes:
        if coord_mode not in cached_angles:
            _, geo_by_date, helio_by_date = cached_bodies[coord_mode]
            bodies_by_date = geo_by_date if geo_by_date else helio_by_date
            
            print(f"  → Углы для {coord_mode.upper()}...")
            cached_angles[coord_mode] = precompute_angles_for_dates(
                bodies_by_date, progress=True
            )
    
    print("✅ Углы закэшированы! Теперь аспекты считаются ~3-5x быстрее.")

    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 6: Запускаем перебор всех комбинаций
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🚀 НАЧИНАЕМ ПЕРЕБОР КОМБИНАЦИЙ")
    print("=" * 80)
    
    results = []
    
    # Track best result so far
    best_so_far = {
        "score": -1.0,
        "gap": 1.0,
        "combo": None,
        "metrics": {}
    }

    # ─────────────────────────────────────────────────────────────────────────────
    # WORKER FUNCTION для параллельного выполнения
    # ─────────────────────────────────────────────────────────────────────────────
    def _evaluate_one_combo(combo_data):
        """Evaluate single combo - used for parallel execution."""
        idx, coord_mode, orb, gw, gs, excl = combo_data
        
        excl_str = f"-[{len(excl)}]" if excl else ""
        if excl and len(excl) <= 2:
            excl_str = f"-[{','.join(excl)}]"
        params_str = f"[{idx+1}/{len(combos)}] {coord_mode} | O={orb} W={gw} S={gs} {excl_str}"
        
        try:
            df_bodies, geo_by_date, helio_by_date = cached_bodies[coord_mode]
            bodies_by_date = geo_by_date if geo_by_date else helio_by_date
            
            res = evaluate_combo(
                df_market, df_bodies, bodies_by_date, settings,
                orb, gw, gs,
                exclude_bodies=excl if excl else None,
                angles_cache=cached_angles.get(coord_mode),
                device=device,
                model_params=config.model_params,
            )
            res["coord_mode"] = coord_mode
            return idx, params_str, excl_str, res, None
        except Exception as e:
            return idx, params_str, excl_str, {
                "coord_mode": coord_mode, "orb_mult": orb, "gauss_window": gw, 
                "gauss_std": gs, "exclude_bodies": excl, "error": str(e)
            }, str(e)

    # ─────────────────────────────────────────────────────────────────────────────
    # PARALLEL or SEQUENTIAL execution
    # ─────────────────────────────────────────────────────────────────────────────
    combo_data_list = [(i, *combo) for i, combo in enumerate(combos)]
    
    if n_workers > 1 or n_workers == -1:
        # PARALLEL EXECUTION with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        actual_workers = n_workers if n_workers > 0 else os.cpu_count() or 4
        print(f"\n⚡ ПАРАЛЛЕЛЬНОЕ ВЫПОЛНЕНИЕ: {actual_workers} потоков")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_evaluate_one_combo, cd): cd for cd in combo_data_list}
            
            for future in as_completed(futures):
                idx, params_str, excl_str, res, error = future.result()
                results.append(res)
                
                if error is None and "error" not in res:
                    r_min = res['recall_min']
                    r_gap = res['recall_gap']
                    mcc = res.get('mcc', 0)
                    coord_mode = res.get('coord_mode', '?')
                    orb = res.get('orb_mult', 0)
                    gw = res.get('gauss_window', 0)
                    gs = res.get('gauss_std', 0)
                    
                    is_best = r_min > best_so_far["score"] or (
                        r_min == best_so_far["score"] and r_gap < best_so_far["gap"]
                    )
                    if is_best:
                        best_so_far["score"] = r_min
                        best_so_far["gap"] = r_gap
                        best_so_far["combo"] = f"{coord_mode} O={orb} W={gw} S={gs} {excl_str}"
                        best_so_far["metrics"] = f"R_MIN={r_min:.3f} GAP={r_gap:.3f} MCC={mcc:.3f}"
                    
                    msg = f"{params_str:<60} → R_UP={res['recall_up']:.3f} R_DOWN={res['recall_down']:.3f} MCC={mcc:.3f}"
                    print(msg)
                    print(f"   🏆 BEST: {best_so_far['metrics']} ({best_so_far['combo']})")
                    print()
                else:
                    print(f"{params_str:<60} → ERROR: {error or res.get('error')}")
                
                # Checkpoint every 100
                if len(results) % 100 == 0:
                    try:
                        ckpt_dir = cfg.reports_dir / "checkpoints"
                        ckpt_dir.mkdir(exist_ok=True, parents=True)
                        ckpt_path = ckpt_dir / f"grid_search_{run_timestamp}_checkpoint.parquet"
                        pd.DataFrame(results).to_parquet(ckpt_path, index=False)
                        print(f"   💾 Checkpoint: {len(results)} combos saved")
                    except Exception as e:
                        print(f"   ⚠️ Checkpoint error: {e}")
    else:
        # SEQUENTIAL EXECUTION (original behavior)
        for i, (coord_mode, orb, gw, gs, excl) in enumerate(combos):
            excl_str = f"-[{len(excl)}]" if excl else ""
            if excl and len(excl) <= 2:
                excl_str = f"-[{','.join(excl)}]"
                
            params_str = f"[{i+1}/{len(combos)}] {coord_mode} | O={orb} W={gw} S={gs} {excl_str}"
            
            try:
                df_bodies, geo_by_date, helio_by_date = cached_bodies[coord_mode]
                bodies_by_date = geo_by_date if geo_by_date else helio_by_date
                
                res = evaluate_combo(
                    df_market, df_bodies, bodies_by_date, settings,
                    orb, gw, gs,
                    exclude_bodies=excl if excl else None,
                    angles_cache=cached_angles.get(coord_mode),
                    device=device,
                    model_params=config.model_params,
                )
                res["coord_mode"] = coord_mode
                results.append(res)
                
                if "error" not in res:
                    r_min = res['recall_min']
                    r_gap = res['recall_gap']
                    mcc = res.get('mcc', 0)
                    
                    is_best = False
                    if r_min > best_so_far["score"]:
                        is_best = True
                    elif r_min == best_so_far["score"] and r_gap < best_so_far["gap"]:
                        is_best = True
                        
                    if is_best:
                        best_so_far["score"] = r_min
                        best_so_far["gap"] = r_gap
                        best_so_far["combo"] = f"{coord_mode} O={orb} W={gw} S={gs} {excl_str}"
                        best_so_far["metrics"] = f"R_MIN={r_min:.3f} GAP={r_gap:.3f} MCC={mcc:.3f}"
                    
                    msg = f"{params_str:<60} → R_UP={res['recall_up']:.3f} R_DOWN={res['recall_down']:.3f} MCC={mcc:.3f}"
                    print(msg)
                    print(f"   🏆 BEST: {best_so_far['metrics']} ({best_so_far['combo']})")
                    print()
                else:
                    print(f"{params_str:<60} → ERROR: {res.get('error')}")

                # CHECKPOINT: Save every 100 iterations
                if (i + 1) % 100 == 0:
                    try:
                        ckpt_dir = cfg.reports_dir / "checkpoints"
                        ckpt_dir.mkdir(exist_ok=True, parents=True)
                        ckpt_path = ckpt_dir / f"grid_search_{run_timestamp}_checkpoint.parquet"
                        pd.DataFrame(results).to_parquet(ckpt_path, index=False)
                        print(f"   💾 Checkpoint saved: {ckpt_path.name}")
                    except Exception as e:
                        print(f"   ⚠️ Checkpoint error: {e}")

            except Exception as e:
                print(f"{params_str:<60} → CRASH: {e}")
                results.append({
                    "coord_mode": coord_mode, "orb_mult": orb, "gauss_window": gw, "gauss_std": gs,
                    "exclude_bodies": excl, "error": str(e)
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    # Sort by QUALITY first, then BALANCE: maximize recall_min, then minimize recall_gap
    if "recall_down" in results_df.columns and "recall_up" in results_df.columns:
        results_df = results_df.sort_values(
            ["recall_min", "recall_gap", "bal_acc"],
            ascending=[False, True, False]
        ).reset_index(drop=True)
    
    # Save results
    if save_results:
        save_grid_search_results(results_df)
    
    # Print best
    print("\n" + "=" * 60)
    print("TOP 5 COMBOS BY BALANCE:")
    print(results_df.head(5).to_string(index=False))
    
    return results_df


def save_grid_search_results(results_df: pd.DataFrame) -> Path:
    """Save grid search results to reports directory."""
    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"grid_search_{timestamp}.csv"
    
    results_df.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")
    
    return path


def get_best_params(results_df: pd.DataFrame) -> Dict:
    """Extract best parameters from grid search results."""
    if results_df.empty:
        return {}
    
    best = results_df.iloc[0].to_dict()
    return {
        "orb_mult": float(best.get("orb_mult", 1.0)),
        "gauss_window": int(best.get("gauss_window", 201)),
        "gauss_std": float(best.get("gauss_std", 50.0)),
        "threshold": float(best.get("threshold", 0.5)),
    }


def save_best_params(params: Dict, name: str = "best") -> Path:
    """Save best parameters to YAML file."""
    import yaml
    
    reports_dir = cfg.reports_dir
    path = reports_dir / f"{name}_params.yaml"
    
    with open(path, "w") as f:
        yaml.dump(params, f) 
    
    print(f"Best params saved to: {path}")
    return path


# =============================================================================
# BODY ABLATION SEARCH
# =============================================================================

def get_all_body_names(settings: Any) -> List[str]:
    """Get list of all body names from settings."""
    return [b.name for b in settings.bodies]


def evaluate_body_exclusion(
    df_market: pd.DataFrame,
    df_bodies: pd.DataFrame,
    df_aspects: pd.DataFrame,
    df_labels: pd.DataFrame,
    exclude_bodies: List[str],
    device: str = "cpu",
    model_params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate model performance when excluding specific bodies.
    
    Args:
        df_market: Market data
        df_bodies: Pre-calculated body positions
        df_aspects: Pre-calculated aspects
        df_labels: Pre-created labels
        exclude_bodies: List of body names to exclude
        device: 'cpu' or 'cuda'
        model_params: XGBoost parameters
    
    Returns:
        Dictionary with exclusion params and metrics
    """
    try:
        # Build features with body exclusion
        df_features = build_full_features(
            df_bodies,
            df_aspects,
            df_transits=None,
            include_pair_aspects=True,
            include_transit_aspects=False,
            exclude_bodies=exclude_bodies,
        )
        
        # Merge with labels
        df_dataset = merge_features_with_labels(df_features, df_labels)
        
        if len(df_dataset) < 100:
            return {"exclude_bodies": exclude_bodies, "error": "Too few samples"}
        
        # Split
        train_df, val_df, test_df = split_dataset(df_dataset)
        
        feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
        X_train, y_train = prepare_xy(train_df, feature_cols)
        X_val, y_val = prepare_xy(val_df, feature_cols)
        X_test, y_test = prepare_xy(test_df, feature_cols)
        
        # Train
        params = model_params or {
            "n_estimators": 300,  # Faster for grid search
            "max_depth": 3,
            "learning_rate": 0.05,
            "verbosity": 0,
        }
        
        model = train_xgb_model(
            X_train, y_train, X_val, y_val,
            feature_cols, n_classes=2, device=device,
            **params
        )
        
        # Tune threshold
        best_t, _ = tune_threshold(model, X_val, y_val, metric="bal_acc")
        
        # Predict on test
        y_pred = predict_with_threshold(model, X_test, threshold=best_t)
        
        # Metrics
        metrics = calc_metrics(y_test, y_pred, [0, 1])
        
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
            "exclude_bodies": exclude_bodies,
            "n_excluded": len(exclude_bodies),
            "n_features": len(feature_cols),
            "n_samples": len(df_dataset),
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
            "mcc": metrics["mcc"],
            "bal_acc": metrics["bal_acc"],
        }
    except Exception as e:
        return {"exclude_bodies": exclude_bodies, "error": str(e)}


def run_body_ablation_search(
    df_market: pd.DataFrame,
    orb_mult: float = 1.0,
    gauss_window: int = 201,
    gauss_std: float = 50.0,
    max_exclude: int = 3,
    n_workers: Optional[int] = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run ablation study on astro bodies to find most influential ones.
    
    Tries all combinations of excluding 1, 2, ... max_exclude bodies
    and measures impact on model performance.
    
    Args:
        df_market: Market data DataFrame
        orb_mult: Orb multiplier to use
        gauss_window: Gaussian window for labeling
        gauss_std: Gaussian std for labeling  
        max_exclude: Maximum number of bodies to exclude (1, 2, 3...)
        n_workers: Number of parallel workers (default: CPU count - 1)
        save_results: Save results to reports directory
    
    Returns:
        DataFrame with all results sorted by balanced accuracy
    """
    from itertools import combinations
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    
    print("=" * 60)
    print("BODY ABLATION SEARCH")
    print("=" * 60)
    
    # Check CUDA
    _, device = check_cuda_available()
    print(f"Device: {device}")
    
    # Workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    print(f"Workers: {n_workers}")
    
    # Initialize astro
    settings = init_ephemeris()
    all_bodies = get_all_body_names(settings)
    print(f"Bodies: {all_bodies}")
    
    # Calculate bodies once
    print("\nCalculating body positions...")
    df_bodies, bodies_by_date = calculate_bodies_for_dates(
        df_market["date"], settings, progress=True
    )
    print(f"  df_bodies.shape: {df_bodies.shape}")
    
    # Calculate aspects once with specified orb
    print(f"\nCalculating aspects (orb_mult={orb_mult})...")
    df_aspects = calculate_aspects_for_dates(
        bodies_by_date, settings, orb_mult=orb_mult, progress=True
    )
    print(f"  df_aspects.shape: {df_aspects.shape}")
    
    # Create labels once
    print(f"\nCreating labels (window={gauss_window}, std={gauss_std})...")
    df_labels = create_balanced_labels(
        df_market,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
    )
    print(f"  df_labels.shape: {df_labels.shape}")
    
    # Generate all exclusion combinations
    exclusion_combos = []
    
    # Baseline: no exclusion
    exclusion_combos.append([])
    
    # 1, 2, ... max_exclude bodies
    for n_exclude in range(1, max_exclude + 1):
        for combo in combinations(all_bodies, n_exclude):
            exclusion_combos.append(list(combo))
    
    print(f"\nTotal combinations to test: {len(exclusion_combos)}")
    print(f"  - Baseline: 1")
    for n in range(1, max_exclude + 1):
        count = len(list(combinations(all_bodies, n)))
        print(f"  - Exclude {n}: {count}")
    
    # Run search (sequential for now - multiprocessing has pickle issues with XGBoost)
    print("\nRunning ablation search...")
    results = []
    
    for i, exclude in enumerate(exclusion_combos):
        exclude_str = ", ".join(exclude) if exclude else "BASELINE"
        print(f"[{i+1}/{len(exclusion_combos)}] Exclude: {exclude_str}", end=" ")
        
        res = evaluate_body_exclusion(
            df_market, df_bodies, df_aspects, df_labels,
            exclude_bodies=exclude,
            device=device,
        )
        results.append(res)
        
        if "error" not in res:
            print(f"→ RECALL_MIN={res['recall_min']:.3f} | RECALL_GAP={res['recall_gap']:.3f} | bal_acc={res['bal_acc']:.3f}")
        else:
            print(f"→ ERROR: {res['error']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add readable exclude column
    results_df["exclude_str"] = results_df["exclude_bodies"].apply(
        lambda x: ", ".join(x) if x else "BASELINE"
    )
    
    # Sort by QUALITY first, then BALANCE: maximize recall_min, then minimize recall_gap
    if "recall_gap" in results_df.columns:
        results_df = results_df.sort_values(
            ["recall_min", "recall_gap", "bal_acc"],
            ascending=[False, True, False]
        ).reset_index(drop=True)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = cfg.reports_dir / f"body_ablation_{timestamp}.csv"
        results_df.to_csv(path, index=False)
        print(f"\nResults saved to: {path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 COMBINATIONS BY BALANCED ACCURACY:")
    print("=" * 60)
    top_cols = ["exclude_str", "n_excluded", "n_features", "bal_acc", "f1_min", "f1_down", "f1_up"]
    available_cols = [c for c in top_cols if c in results_df.columns]
    print(results_df[available_cols].head(10).to_string(index=False))
    
    # Identify most influential bodies
    print("\n" + "=" * 60)
    print("BODY INFLUENCE ANALYSIS:")
    print("=" * 60)
    
    baseline = results_df[results_df["exclude_str"] == "BASELINE"]
    if not baseline.empty:
        baseline_acc = baseline.iloc[0]["bal_acc"]
        print(f"Baseline accuracy: {baseline_acc:.3f}")
        
        # For single-body exclusions, show impact
        single_exclusions = results_df[results_df["n_excluded"] == 1].copy()
        if not single_exclusions.empty:
            single_exclusions["impact"] = baseline_acc - single_exclusions["bal_acc"]
            single_exclusions = single_exclusions.sort_values("impact", ascending=False)
            
            print("\nImpact of removing each body (positive = body helps model):")
            for _, row in single_exclusions.iterrows():
                impact = row["impact"]
                sign = "+" if impact > 0 else ""
                body = row["exclude_str"]
                print(f"  {body:12s}: {sign}{impact:.3f}")
    
    return results_df


def run_comprehensive_search(
    df_market: pd.DataFrame,
    orb_multipliers: List[float] = [0.5, 0.8, 1.0, 1.2],
    gauss_windows: List[int] = [101, 151, 201],
    gauss_stds: List[float] = [30.0, 50.0, 70.0],
    max_exclude_bodies: int = 2,
    save_results: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive grid search: first gauss+orb, then body ablation with best params.
    
    Args:
        df_market: Market data
        orb_multipliers: List of orb values to try
        gauss_windows: List of gaussian window sizes
        gauss_stds: List of gaussian stds
        max_exclude_bodies: Max bodies to exclude in ablation
        save_results: Save results to reports
    
    Returns:
        Dictionary with 'gauss_orb' and 'ablation' DataFrames
    """
    print("=" * 70)
    print("COMPREHENSIVE GRID SEARCH")
    print("=" * 70)
    
    # Phase 1: Find best gauss + orb
    print("\n" + "=" * 70)
    print("PHASE 1: GAUSSIAN + ORB PARAMETER SEARCH")
    print("=" * 70)
    
    config = GridSearchConfig(
        orb_multipliers=orb_multipliers,
        gauss_windows=gauss_windows,
        gauss_stds=gauss_stds,
    )
    
    gauss_orb_results = run_grid_search(df_market, config, save_results=save_results)
    best_params = get_best_params(gauss_orb_results)
    
    print(f"\nBest params from Phase 1:")
    print(f"  orb_mult: {best_params['orb_mult']}")
    print(f"  gauss_window: {best_params['gauss_window']}")
    print(f"  gauss_std: {best_params['gauss_std']}")
    
    # Phase 2: Body ablation with best params
    print("\n" + "=" * 70)
    print("PHASE 2: BODY ABLATION SEARCH")
    print("=" * 70)
    
    ablation_results = run_body_ablation_search(
        df_market,
        orb_mult=best_params["orb_mult"],
        gauss_window=best_params["gauss_window"],
        gauss_std=best_params["gauss_std"],
        max_exclude=max_exclude_bodies,
        save_results=save_results,
    )
    
    return {
        "gauss_orb": gauss_orb_results,
        "ablation": ablation_results,
        "best_params": best_params,
    }


def run_full_grid_search(
    df_market: pd.DataFrame,
    orb_multipliers: List[float] = [0.5, 0.8, 1.0, 1.2],
    gauss_windows: List[int] = [101, 151, 201],
    gauss_stds: List[float] = [30.0, 50.0, 70.0],
    test_mode: bool = True,
    test_limit: int = 20,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run FULL grid search over ALL combinations: gauss × orb × body exclusions.
    
    WARNING: Full search is VERY slow! Use test_mode=True first.
    
    Full search space:
    - gauss: 3 × orb: 4 = 12 param combos
    - bodies: 2^13 = 8192 exclusion combos
    - Total: ~98,000 combinations
    
    Args:
        df_market: Market data
        orb_multipliers: List of orb values
        gauss_windows: List of gaussian windows
        gauss_stds: List of gaussian stds
        test_mode: If True, run only test_limit combinations
        test_limit: Number of combos to test in test_mode
        save_results: Save to reports directory
    
    Returns:
        DataFrame with all results
    """
    from itertools import combinations, product
    import time
    
    print("=" * 70)
    print("FULL GRID SEARCH: GAUSS × ORB × BODY EXCLUSIONS")
    print("=" * 70)
    
    # Check CUDA
    _, device = check_cuda_available()
    print(f"Device: {device}")
    
    # Initialize astro
    settings = init_ephemeris()
    all_bodies = get_all_body_names(settings)
    n_bodies = len(all_bodies)
    print(f"Bodies ({n_bodies}): {all_bodies}")
    
    # Calculate bodies once
    print("\nCalculating body positions (one-time)...")
    df_bodies, bodies_by_date = calculate_bodies_for_dates(
        df_market["date"], settings, progress=True
    )
    print(f"  df_bodies.shape: {df_bodies.shape}")
    
    # Generate ALL body exclusion combinations (2^n)
    all_exclusions = [[]]  # Start with no exclusion
    for r in range(1, n_bodies + 1):
        for combo in combinations(all_bodies, r):
            all_exclusions.append(list(combo))
    
    print(f"\nTotal body exclusion combinations: {len(all_exclusions)}")
    
    # Generate all param combinations
    param_combos = list(product(orb_multipliers, gauss_windows, gauss_stds))
    print(f"Param combinations (orb × gauss): {len(param_combos)}")
    
    # Generate full grid
    full_grid = []
    for orb, gw, gs in param_combos:
        for exclude in all_exclusions:
            full_grid.append({
                "orb_mult": orb,
                "gauss_window": gw,
                "gauss_std": gs,
                "exclude_bodies": exclude,
            })
    
    total_combos = len(full_grid)
    print(f"\nTOTAL COMBINATIONS: {total_combos:,}")
    
    if test_mode:
        print(f"\n*** TEST MODE: Running only {test_limit} combinations ***")
        # Sample diverse combinations
        import random
        random.seed(42)
        full_grid = random.sample(full_grid, min(test_limit, total_combos))
    
    print(f"\nRunning {len(full_grid)} combinations...")
    
    # Pre-compute aspects for each orb (cache)
    aspects_cache = {}
    unique_orbs = list(set(c["orb_mult"] for c in full_grid))
    print(f"\nPre-computing aspects for {len(unique_orbs)} orb values...")
    for orb in unique_orbs:
        df_aspects = calculate_aspects_for_dates(
            bodies_by_date, settings, orb_mult=orb, progress=False
        )
        aspects_cache[orb] = df_aspects
        print(f"  orb={orb}: {len(df_aspects)} aspects")
    
    # Pre-compute labels for each gauss config (cache)
    labels_cache = {}
    unique_gauss = list(set((c["gauss_window"], c["gauss_std"]) for c in full_grid))
    print(f"\nPre-computing labels for {len(unique_gauss)} gauss configs...")
    for gw, gs in unique_gauss:
        df_labels = create_balanced_labels(df_market, gauss_window=gw, gauss_std=gs)
        labels_cache[(gw, gs)] = df_labels
        print(f"  window={gw}, std={gs}: {len(df_labels)} labels")
    
    # Run search
    print("\n" + "=" * 70)
    print("RUNNING GRID SEARCH...")
    print("=" * 70)
    
    results = []
    start_time = time.time()
    
    for i, combo in enumerate(full_grid):
        orb = combo["orb_mult"]
        gw = combo["gauss_window"]
        gs = combo["gauss_std"]
        exclude = combo["exclude_bodies"]
        
        exclude_str = ", ".join(exclude) if exclude else "NONE"
        
        # Get cached data
        df_aspects = aspects_cache[orb]
        df_labels = labels_cache[(gw, gs)]
        
        # Progress
        elapsed = time.time() - start_time
        if i > 0:
            avg_per_combo = elapsed / i
            remaining = avg_per_combo * (len(full_grid) - i)
            eta_min = remaining / 60
        else:
            eta_min = 0
        
        print(f"[{i+1}/{len(full_grid)}] orb={orb}, gw={gw}, gs={gs}, exclude=[{exclude_str[:30]}...]", end=" ")
        
        # Evaluate
        res = evaluate_body_exclusion(
            df_market, df_bodies, df_aspects, df_labels,
            exclude_bodies=exclude,
            device=device,
        )
        
        # Add params to result
        res["orb_mult"] = orb
        res["gauss_window"] = gw
        res["gauss_std"] = gs
        res["exclude_str"] = exclude_str
        res["n_excluded"] = len(exclude)
        
        results.append(res)
        
        if "error" not in res:
            print(f"→ RECALL_MIN={res['recall_min']:.3f} | RECALL_GAP={res['recall_gap']:.3f} | mcc={res.get('mcc', 0):.3f} (ETA: {eta_min:.1f}min)")
        else:
            print(f"→ ERROR")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by QUALITY first, then BALANCE: maximize recall_min, then minimize recall_gap
    if "recall_gap" in results_df.columns:
        results_df = results_df.sort_values(
            ["recall_min", "recall_gap", "bal_acc"],
            ascending=[False, True, False]
        ).reset_index(drop=True)
    
    # Save
    if save_results:
        mode_str = "test" if test_mode else "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = cfg.reports_dir / f"full_grid_{mode_str}_{timestamp}.csv"
        results_df.to_csv(path, index=False)
        print(f"\nResults saved to: {path}")
    
    # Print top results
    print("\n" + "=" * 70)
    print("TOP 15 COMBINATIONS:")
    print("=" * 70)
    top_cols = ["orb_mult", "gauss_window", "gauss_std", "exclude_str", "n_excluded", "bal_acc", "f1_min"]
    available = [c for c in top_cols if c in results_df.columns]
    print(results_df[available].head(15).to_string(index=False))
    
    return results_df


def evaluate_and_plot_best(
    df_market: pd.DataFrame,
    best_row: pd.Series,
) -> Dict:
    """
    Fully evaluate and visualize the best grid search result.
    
    Args:
        df_market: Market data
        best_row: Row from grid search results DataFrame (results_df.iloc[0])
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    from .visualization import plot_predictions, plot_confusion_matrix
    from sklearn.metrics import classification_report
    
    print("=" * 70)
    print("EVALUATING BEST GRID SEARCH RESULT")
    print("=" * 70)
    
    # Extract params
    orb_mult = float(best_row.get("orb_mult", 1.0))
    gauss_window = int(best_row.get("gauss_window", 201))
    gauss_std = float(best_row.get("gauss_std", 50.0))
    
    # Handle exclude_bodies - can be list or string
    exclude_bodies = best_row.get("exclude_bodies", [])
    if isinstance(exclude_bodies, str):
        exclude_bodies = [b.strip() for b in exclude_bodies.split(",") if b.strip() and b.strip() != "NONE"]
    
    print(f"\nBest params:")
    print(f"  orb_mult: {orb_mult}")
    print(f"  gauss_window: {gauss_window}")
    print(f"  gauss_std: {gauss_std}")
    print(f"  exclude_bodies: {exclude_bodies if exclude_bodies else 'NONE'}")
    
    # Check CUDA
    _, device = check_cuda_available()
    print(f"  device: {device}")
    
    # Initialize astro
    settings = init_ephemeris()
    
    # Calculate bodies
    print("\nCalculating body positions...")
    df_bodies, bodies_by_date = calculate_bodies_for_dates(
        df_market["date"], settings, progress=True
    )
    
    # Calculate aspects
    print(f"\nCalculating aspects (orb_mult={orb_mult})...")
    df_aspects = calculate_aspects_for_dates(
        bodies_by_date, settings, orb_mult=orb_mult, progress=True
    )
    
    # Create labels
    print(f"\nCreating labels (window={gauss_window}, std={gauss_std})...")
    df_labels = create_balanced_labels(
        df_market,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
    )
    
    # Build features with exclusion
    print(f"\nBuilding features (exclude: {exclude_bodies if exclude_bodies else 'NONE'})...")
    df_features = build_full_features(
        df_bodies, df_aspects,
        exclude_bodies=exclude_bodies if exclude_bodies else None,
    )
    
    # Merge with labels
    df_dataset = merge_features_with_labels(df_features, df_labels)
    print(f"Dataset shape: {df_dataset.shape}")
    
    # Split
    train_df, val_df, test_df = split_dataset(df_dataset)
    feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
    
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Train model
    print("\nTraining model...")
    model = train_xgb_model(
        X_train, y_train, X_val, y_val,
        feature_cols, n_classes=2, device=device,
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
    )
    
    # Tune threshold
    best_t, _ = tune_threshold(model, X_val, y_val, metric="bal_acc")
    print(f"\nBest threshold: {best_t:.3f}")
    
    # Predict
    y_pred = predict_with_threshold(model, X_test, threshold=best_t)
    
    # Full evaluation
    print("\n" + "=" * 80)
    print("FULL EVALUATION ON TEST SET")
    print("=" * 80)
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ЧТО ЗДЕСЬ ПРОИСХОДИТ:                                                       ║
║  Модель обучена на ПРОШЛЫХ данных и теперь предсказывает БУДУЩИЕ движения.   ║
║  Тестовый набор - это данные, которые модель НИКОГДА НЕ ВИДЕЛА при обучении. ║
║  Это честная проверка: сможет ли модель предсказывать на новых данных?       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    metrics = calc_metrics(y_test, y_pred, [0, 1])
    
    # Get per-class metrics from classification report
    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["DOWN", "UP"], output_dict=True, zero_division=0
    )
    
    f1_down = report_dict["DOWN"]["f1-score"]
    f1_up = report_dict["UP"]["f1-score"]
    f1_min = min(f1_down, f1_up)
    f1_gap = abs(f1_down - f1_up)
    
    recall_down = report_dict["DOWN"]["recall"]
    recall_up = report_dict["UP"]["recall"]
    recall_min = min(recall_down, recall_up)
    recall_gap = abs(recall_down - recall_up)
    
    precision_down = report_dict["DOWN"]["precision"]
    precision_up = report_dict["UP"]["precision"]
    
    # Count predictions
    n_pred_down = int((y_pred == 0).sum())
    n_pred_up = int((y_pred == 1).sum())
    n_true_down = int((y_test == 0).sum())
    n_true_up = int((y_test == 1).sum())
    
    # Correct predictions
    correct_down = int(((y_pred == 0) & (y_test == 0)).sum())  # True Negatives
    correct_up = int(((y_pred == 1) & (y_test == 1)).sum())    # True Positives
    
    print("=" * 80)
    print("📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
    print("=" * 80)
    print(f"""
  В тестовом наборе {len(y_test)} дней:
    • Реально было DOWN (падение): {n_true_down} дней
    • Реально было UP (рост):      {n_true_up} дней
    
  Модель предсказала:
    • DOWN (падение): {n_pred_down} раз
    • UP (рост):      {n_pred_up} раз
    
  Правильных предсказаний:
    • DOWN→DOWN (правильно угадали падение): {correct_down} из {n_true_down} = {recall_down*100:.1f}%
    • UP→UP (правильно угадали рост):        {correct_up} из {n_true_up} = {recall_up*100:.1f}%
""")
    
    print("=" * 80)
    print("🎯 ГЛАВНЫЕ МЕТРИКИ (ЧЕМ ВЫШЕ - ТЕМ ЛУЧШЕ):")
    print("=" * 80)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ RECALL (Полнота) - какой % реальных событий модель поймала:                │
  │   • RECALL DOWN = {recall_down:.1%}  (из всех реальных падений угадали {recall_down:.1%})     │
  │   • RECALL UP   = {recall_up:.1%}  (из всех реальных ростов угадали {recall_up:.1%})       │
  │   • RECALL MIN  = {recall_min:.1%}  ← ХУДШИЙ КЛАСС (главная метрика!)            │
  │   • RECALL GAP  = {recall_gap:.1%}  ← разница между классами (чем меньше = лучше) │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ PRECISION (Точность) - какой % предсказаний оказался верным:                │
  │   • PRECISION DOWN = {precision_down:.1%}  (из предсказанных DOWN верных {precision_down:.1%})    │
  │   • PRECISION UP   = {precision_up:.1%}  (из предсказанных UP верных {precision_up:.1%})       │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ ОБЩИЕ МЕТРИКИ:                                                              │
  │   • BALANCED ACCURACY = {metrics['bal_acc']:.1%}  (среднее recall обоих классов)      │
  │   • MCC = {metrics['mcc']:+.3f}  (от -1 до +1, где 0 = случайное угадывание)       │
  │   • F1 MACRO = {metrics['f1_macro']:.1%}  (гармоническое среднее precision и recall)│
  └─────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Quality assessment
    print("=" * 80)
    print("📋 ОЦЕНКА КАЧЕСТВА:")
    print("=" * 80)
    
    if recall_min > 0.55:
        quality = "✅ ХОРОШО"
        quality_msg = "Модель предсказывает оба класса лучше случайного!"
    elif recall_min > 0.50:
        quality = "⚠️ СРЕДНЕ"
        quality_msg = "Модель чуть лучше случайного угадывания."
    else:
        quality = "❌ ПЛОХО"
        quality_msg = "Модель хуже случайного угадывания для одного из классов!"
    
    if recall_gap < 0.10:
        balance = "✅ СБАЛАНСИРОВАНО"
        balance_msg = "Модель одинаково хорошо предсказывает оба класса."
    elif recall_gap < 0.20:
        balance = "⚠️ НЕБОЛЬШОЙ ДИСБАЛАНС"
        balance_msg = "Модель немного лучше предсказывает один класс."
    else:
        balance = "❌ СИЛЬНЫЙ ДИСБАЛАНС"
        balance_msg = "Модель сильно смещена к одному классу!"
    
    print(f"""
  КАЧЕСТВО:  {quality}
  → {quality_msg}
  
  БАЛАНС:    {balance}
  → {balance_msg}
  
  ВЕРДИКТ:   {"Модель можно использовать!" if recall_min > 0.52 and recall_gap < 0.15 else "Требуется улучшение параметров."}
""")
    
    print("=" * 80)
    print("📋 ПОЛНЫЙ ОТЧЕТ SKLEARN (Classification Report):")
    print("=" * 80)
    print("""
  Что означают колонки:
    • precision - точность: из всех предсказаний класса, сколько % верных
    • recall    - полнота: из всех реальных примеров класса, сколько % нашли  
    • f1-score  - гармоническое среднее precision и recall
    • support   - количество реальных примеров этого класса в тесте
""")
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["DOWN", "UP"], zero_division=0
    )
    print(report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    plot_confusion_matrix(y_test, y_pred)
    
    # Predictions plot
    print("\nPredictions vs True Labels:")
    
    # Prepare test_df with close prices
    test_df_plot = test_df.copy()
    test_df_plot["date"] = pd.to_datetime(test_df_plot["date"])
    market_close = df_market[["date", "close"]].copy()
    market_close["date"] = pd.to_datetime(market_close["date"])
    test_df_plot = test_df_plot.merge(market_close, on="date", how="left")
    
    plot_predictions(test_df_plot, y_pred, y_true=y_test, price_mode="log")
    
    return {
        "model": model,
        "threshold": best_t,
        "y_pred": y_pred,
        "y_test": y_test,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "params": {
            "orb_mult": orb_mult,
            "gauss_window": gauss_window,
            "gauss_std": gauss_std,
            "exclude_bodies": exclude_bodies,
        },
    }
