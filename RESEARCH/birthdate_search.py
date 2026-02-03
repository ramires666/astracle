# %% [markdown]
# # 🎂 Bitcoin Birth Date Search
# 
# Поиск оптимальной даты рождения Bitcoin через перебор всех 365 дней года.
# 
# ## Идея:
# - Bitcoin был создан в 2009 году, но точная "дата рождения" неизвестна
# - Возможные варианты: genesis block (03.01.2009), whitepaper (31.10.2008), etc.
# - Мы перебираем ВСЕ 365 дней и смотрим какая дата даёт лучшую модель
# 
# ## Фаза 1: Только транзиты
# - Используем ТОЛЬКО транзиты к натальной карте субъекта
# - НЕ используем аспекты между транзитными планетами
# 
# ## Фаза 2: Транзиты + лучший baseline
# - Добавляем транзиты к лучшей модели из hyperparameter search

# %% [markdown]
# ## 1. Setup

# %%
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/home/rut/ostrofun")
sys.path.insert(0, str(PROJECT_ROOT))

# %%
# Import all modules
from RESEARCH.config import cfg, PROJECT_ROOT
from RESEARCH.data_loader import load_market_data
from RESEARCH.labeling import create_balanced_labels
from RESEARCH.astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates,
    calculate_bodies_for_dates_multi,
    calculate_aspects_for_dates,
    calculate_transits_for_dates,
    calculate_phases_for_dates,
    get_natal_bodies,
)
from RESEARCH.features import (
    build_full_features,
    merge_features_with_labels,
    get_feature_columns,
)
from RESEARCH.model_training import (
    split_dataset,
    prepare_xy,
    train_xgb_model,
    tune_threshold,
    predict_with_threshold,
    check_cuda_available,
)
from RESEARCH.grid_search import evaluate_combo
from RESEARCH.visualization import plot_confusion_matrix

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from itertools import product
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("✓ All modules imported")

# %% [markdown]
# ## 2. Configuration

# %%
# ══════════════════════════════════════════════════════════════════════════════════
# 🎂 НАСТРОЙКИ ПОИСКА ДАТЫ РОЖДЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════════

# Год рождения Bitcoin (перебираем все дни этого года)
BIRTH_YEAR = 2009  # Genesis block год

# Диапазон дат для поиска
# Можно ограничить если есть предположения
SEARCH_START = date(BIRTH_YEAR, 1, 1)   # С 1 января
SEARCH_END = date(BIRTH_YEAR, 12, 31)    # По 31 декабря

# Известные даты-кандидаты Bitcoin (для приоритетной проверки)
CANDIDATE_DATES = [
    date(2009, 1, 3),   # Genesis Block mined
    date(2009, 1, 9),   # First Bitcoin software release
    date(2009, 1, 12),  # First Bitcoin transaction (Satoshi → Hal Finney)
    date(2008, 10, 31), # Whitepaper published
    date(2008, 8, 18),  # bitcoin.org domain registered
]

# ══════════════════════════════════════════════════════════════════════════════════
# 🎯 ЛУЧШИЕ ПАРАМЕТРЫ ИЗ HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════════════
# Эти параметры используются для модели

BEST_ASTRO_CONFIG = {
    "coord_mode": "both",      # geo + helio координаты
    "orb_mult": 0.15,          # Орбис для аспектов
    "gauss_window": 300,       # Гаусс окно для меток
    "gauss_std": 70.0,         # Гаусс std для меток
    "exclude_bodies": None,    # Все тела включены
}

BEST_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
}

print(f"🎂 Search range: {SEARCH_START} → {SEARCH_END}")
print(f"   Total days: {(SEARCH_END - SEARCH_START).days + 1}")
print(f"   Known candidates: {len(CANDIDATE_DATES)}")

# %% [markdown]
# ## 3. Load Data

# %%
# Load market data
df_market = load_market_data()
DATA_START = "2017-11-01"
df_market = df_market[df_market["date"] >= DATA_START].reset_index(drop=True)

print(f"Market data: {len(df_market)} rows")
print(f"Date range: {df_market['date'].min().date()} → {df_market['date'].max().date()}")

# %%
# Create labels
df_labels = create_balanced_labels(
    df_market,
    gauss_window=BEST_ASTRO_CONFIG["gauss_window"],
    gauss_std=BEST_ASTRO_CONFIG["gauss_std"],
)

print(f"Labels: {len(df_labels)} rows")
print(f"Distribution: {df_labels['target'].value_counts().to_dict()}")

# %%
# Initialize ephemeris
settings = init_ephemeris()
_, device = check_cuda_available()
print(f"Device: {device}")

# %% [markdown]
# ## 4. Pre-compute Transit Bodies

# %%
# Вычисляем позиции транзитных планет для всех дат (один раз)
print("📍 Pre-computing transit body positions...")

if BEST_ASTRO_CONFIG["coord_mode"] == "geo":
    df_bodies, bodies_by_date = calculate_bodies_for_dates(
        df_market["date"], settings, progress=True
    )
else:
    df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
        df_market["date"], settings, 
        coord_mode=BEST_ASTRO_CONFIG["coord_mode"], 
        progress=True
    )
    bodies_by_date = geo_by_date if geo_by_date else helio_by_date

print(f"✓ Transit bodies: {len(df_bodies)} rows")

# %%
# Вычисляем фазы Луны и элонгации планет
print("\n🌙 Computing moon phases and elongations...")
df_phases = calculate_phases_for_dates(bodies_by_date, progress=True)
print(f"✓ Phases: {len(df_phases)} rows")

# %% [markdown]
# ## 5. Birth Date Evaluation Function

# %%
def evaluate_birthdate(
    birth_date: date,
    df_market: pd.DataFrame,
    df_labels: pd.DataFrame,
    bodies_by_date: dict,
    df_phases: pd.DataFrame,
    settings,
    config: dict,
    xgb_params: dict,
    device: str,
    use_pair_aspects: bool = False,  # Фаза 1: False, Фаза 2: True
) -> dict:
    """
    Оценка одной даты рождения.
    
    Возвращает метрики модели обученной на транзитах к этой дате.
    """
    try:
        # 1. Получаем натальные позиции для этой даты
        natal_bodies = get_natal_bodies(birth_date, settings)
        
        # 2. Вычисляем транзиты к натальным позициям
        df_transits = calculate_transits_for_dates(
            bodies_by_date, natal_bodies, settings,
            orb_mult=config["orb_mult"],
            progress=False,
        )
        
        if len(df_transits) == 0:
            return {"error": "No transits found", "birth_date": birth_date}
        
        # 3. Вычисляем аспекты между планетами (если Фаза 2)
        df_aspects = None
        if use_pair_aspects:
            df_aspects = calculate_aspects_for_dates(
                bodies_by_date, settings,
                orb_mult=config["orb_mult"],
                progress=False,
            )
        
        # 4. Строим признаки
        df_features = build_full_features(
            df_bodies,
            df_aspects,  # None для Фазы 1
            df_transits=df_transits,  # Транзиты к натальной карте
            df_phases=df_phases,
            include_pair_aspects=use_pair_aspects,
            include_transit_aspects=True,  # Всегда включаем транзиты
            exclude_bodies=config["exclude_bodies"],
        )
        
        # 5. Merge с метками
        df_dataset = merge_features_with_labels(df_features, df_labels)
        
        if len(df_dataset) < 100:
            return {"error": "Too few samples", "birth_date": birth_date}
        
        # 6. Split
        train_df, val_df, test_df = split_dataset(df_dataset)
        feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
        
        X_train, y_train = prepare_xy(train_df, feature_cols)
        X_val, y_val = prepare_xy(val_df, feature_cols)
        X_test, y_test = prepare_xy(test_df, feature_cols)
        
        # 7. Train model
        model = train_xgb_model(
            X_train, y_train, X_val, y_val,
            feature_cols, n_classes=2, device=device,
            **xgb_params
        )
        
        # 8. Tune threshold
        best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min")
        
        # 9. Predict on test
        y_pred = predict_with_threshold(model, X_test, threshold=best_t)
        
        # 10. Metrics
        report = classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["DOWN", "UP"], output_dict=True, zero_division=0
        )
        
        recall_down = report["DOWN"]["recall"]
        recall_up = report["UP"]["recall"]
        mcc = matthews_corrcoef(y_test, y_pred)
        
        return {
            "birth_date": birth_date,
            "recall_down": recall_down,
            "recall_up": recall_up,
            "recall_min": min(recall_down, recall_up),
            "recall_gap": abs(recall_down - recall_up),
            "mcc": mcc,
            "threshold": best_t,
            "n_features": len(feature_cols),
            "n_transits": len(df_transits),
        }
        
    except Exception as e:
        return {"error": str(e), "birth_date": birth_date}

# %% [markdown]
# ## 6. Phase 1: Transits Only Search

# %%
# ══════════════════════════════════════════════════════════════════════════════════
# 🔍 ФАЗА 1: ПОИСК ДАТЫ РОЖДЕНИЯ (ТОЛЬКО ТРАНЗИТЫ)
# ══════════════════════════════════════════════════════════════════════════════════
# 
# Модель обучается ТОЛЬКО на:
# - Транзиты к натальной карте субъекта
# - Фазы Луны и элонгации планет
# - Позиции транзитных планет
#
# НЕ включает:
# - Аспекты между транзитными планетами

print("=" * 80)
print("🎂 PHASE 1: BIRTH DATE SEARCH (TRANSITS ONLY)")
print("=" * 80)
print(f"   Search range: {SEARCH_START} → {SEARCH_END}")
print(f"   Model params: {BEST_XGB_PARAMS}")
print()

# %%
# Сначала проверяем известных кандидатов
print("🎯 Testing known candidate dates first...")
candidate_results = []

for birth_date in CANDIDATE_DATES:
    if not (SEARCH_START <= birth_date <= SEARCH_END or birth_date.year == 2008):
        # Для 2008 года всё равно тестируем
        pass
    
    result = evaluate_birthdate(
        birth_date, df_market, df_labels, bodies_by_date, df_phases,
        settings, BEST_ASTRO_CONFIG, BEST_XGB_PARAMS, device,
        use_pair_aspects=False,  # Фаза 1: только транзиты
    )
    candidate_results.append(result)
    
    if "error" not in result:
        print(f"   {birth_date}: R_MIN={result['recall_min']:.3f} MCC={result['mcc']:.3f}")
    else:
        print(f"   {birth_date}: ERROR - {result['error']}")

# %%
# Полный перебор всех дней года
print(f"\n🔢 Full search: {(SEARCH_END - SEARCH_START).days + 1} days...")

results_phase1 = []
best_so_far = {"recall_min": 0, "birth_date": None}

current_date = SEARCH_START
with tqdm(total=(SEARCH_END - SEARCH_START).days + 1, desc="Birth date search") as pbar:
    while current_date <= SEARCH_END:
        result = evaluate_birthdate(
            current_date, df_market, df_labels, bodies_by_date, df_phases,
            settings, BEST_ASTRO_CONFIG, BEST_XGB_PARAMS, device,
            use_pair_aspects=False,
        )
        results_phase1.append(result)
        
        if "error" not in result:
            r_min = result["recall_min"]
            if r_min > best_so_far["recall_min"]:
                best_so_far["recall_min"] = r_min
                best_so_far["mcc"] = result["mcc"]
                best_so_far["birth_date"] = current_date
                tqdm.write(f"🏆 NEW BEST: {current_date} → R_MIN={r_min:.3f} MCC={result['mcc']:.3f}")
        
        current_date += timedelta(days=1)
        pbar.update(1)

# %%
# Результаты Фазы 1
print("\n" + "=" * 80)
print("📊 PHASE 1 RESULTS: TRANSITS ONLY")
print("=" * 80)

df_results1 = pd.DataFrame([r for r in results_phase1 if "error" not in r])

if len(df_results1) > 0:
    df_results1 = df_results1.sort_values("recall_min", ascending=False)
    
    print("\n🏆 TOP 20 BEST BIRTH DATES:")
    print("-" * 60)
    print(f"{'#':<3} {'Date':<12} {'R_MIN':<8} {'R_UP':<8} {'R_DOWN':<8} {'MCC':<8}")
    print("-" * 60)
    
    for i, row in df_results1.head(20).iterrows():
        print(f"{df_results1.index.get_loc(i)+1:<3} {str(row['birth_date']):<12} "
              f"{row['recall_min']:.3f}    {row['recall_up']:.3f}    {row['recall_down']:.3f}    {row['mcc']:.3f}")
    
    # Сохраняем результаты
    results_path = PROJECT_ROOT / "data" / "market" / "reports" / "birthdate_search_phase1.csv"
    df_results1.to_csv(results_path, index=False)
    print(f"\n💾 Results saved: {results_path}")
    
    BEST_BIRTHDATE = df_results1.iloc[0]["birth_date"]
    print(f"\n🎂 BEST BIRTH DATE: {BEST_BIRTHDATE}")
    print(f"   R_MIN = {df_results1.iloc[0]['recall_min']:.3f}")
    print(f"   MCC   = {df_results1.iloc[0]['mcc']:.3f}")

# %% [markdown]
# ## 7. Phase 2: Transits + Pair Aspects

# %%
# ══════════════════════════════════════════════════════════════════════════════════
# 🔍 ФАЗА 2: ТРАНЗИТЫ + АСПЕКТЫ МЕЖДУ ПЛАНЕТАМИ
# ══════════════════════════════════════════════════════════════════════════════════
#
# Тестируем TOP-10 лучших дат из Фазы 1, но теперь добавляем:
# - Аспекты между транзитными планетами (из лучшей модели)

print("\n" + "=" * 80)
print("🎂 PHASE 2: TRANSITS + PAIR ASPECTS")
print("=" * 80)

# Берём ТОП-10 дат из Фазы 1
top_dates = df_results1.head(10)["birth_date"].tolist()
print(f"Testing top {len(top_dates)} dates with pair aspects added...")

results_phase2 = []

for birth_date in tqdm(top_dates, desc="Phase 2"):
    result = evaluate_birthdate(
        birth_date, df_market, df_labels, bodies_by_date, df_phases,
        settings, BEST_ASTRO_CONFIG, BEST_XGB_PARAMS, device,
        use_pair_aspects=True,  # Фаза 2: добавляем аспекты!
    )
    results_phase2.append(result)

# %%
# Результаты Фазы 2
print("\n" + "=" * 80)
print("📊 PHASE 2 RESULTS: TRANSITS + PAIR ASPECTS")
print("=" * 80)

df_results2 = pd.DataFrame([r for r in results_phase2 if "error" not in r])

if len(df_results2) > 0:
    df_results2 = df_results2.sort_values("recall_min", ascending=False)
    
    print("\n🏆 COMBINED MODEL RESULTS:")
    print("-" * 70)
    print(f"{'Date':<12} {'R_MIN':<8} {'R_UP':<8} {'R_DOWN':<8} {'MCC':<8} {'Features':<10}")
    print("-" * 70)
    
    for _, row in df_results2.iterrows():
        print(f"{str(row['birth_date']):<12} {row['recall_min']:.3f}    {row['recall_up']:.3f}    "
              f"{row['recall_down']:.3f}    {row['mcc']:.3f}    {row['n_features']}")
    
    # Сохраняем результаты
    results_path2 = PROJECT_ROOT / "data" / "market" / "reports" / "birthdate_search_phase2.csv"
    df_results2.to_csv(results_path2, index=False)
    print(f"\n💾 Results saved: {results_path2}")
    
    FINAL_BEST = df_results2.iloc[0]
    print(f"\n🎂 FINAL BEST BIRTH DATE: {FINAL_BEST['birth_date']}")
    print(f"   R_MIN    = {FINAL_BEST['recall_min']:.3f}")
    print(f"   MCC      = {FINAL_BEST['mcc']:.3f}")
    print(f"   Features = {FINAL_BEST['n_features']}")

# %% [markdown]
# ## 8. Visualization

# %%
# График распределения R_MIN по датам
if len(df_results1) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: R_MIN by date
    ax1 = axes[0]
    dates = pd.to_datetime(df_results1["birth_date"])
    ax1.scatter(dates, df_results1["recall_min"], alpha=0.6, s=20)
    
    # Highlight best
    best_row = df_results1.iloc[0]
    ax1.scatter([pd.to_datetime(best_row["birth_date"])], [best_row["recall_min"]], 
                c='red', s=100, zorder=5, label=f"Best: {best_row['birth_date']}")
    
    # Mark known candidates
    for cd in CANDIDATE_DATES:
        if cd.year == BIRTH_YEAR:
            ax1.axvline(pd.to_datetime(cd), color='green', alpha=0.5, linestyle='--')
    
    ax1.set_xlabel("Birth Date")
    ax1.set_ylabel("R_MIN")
    ax1.set_title("Phase 1: R_MIN Distribution by Birth Date")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MCC by date
    ax2 = axes[1]
    ax2.scatter(dates, df_results1["mcc"], alpha=0.6, s=20, c='orange')
    ax2.scatter([pd.to_datetime(best_row["birth_date"])], [best_row["mcc"]], 
                c='red', s=100, zorder=5)
    ax2.set_xlabel("Birth Date")
    ax2.set_ylabel("MCC")
    ax2.set_title("Phase 1: MCC Distribution by Birth Date")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
print("\n" + "=" * 80)
print("✅ BIRTH DATE SEARCH COMPLETE!")
print("=" * 80)
