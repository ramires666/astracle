# %% [markdown]
# # ⏰ Bitcoin Birth TIME Search (with Houses)
# 
# Поиск точного времени рождения Bitcoin для даты **2009-10-10**
# (первый курс обмена BTC/USD).
# 
# ## Включает:
# - Перебор времени с шагом 5 минут (288 вариантов за день)
# - Расчёт домов (куспиды)
# - Аспекты транзитных планет к куспидам домов
# - Позиции планет в домах

# %% [markdown]
# ## 1. Setup

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/rut/ostrofun")
sys.path.insert(0, str(PROJECT_ROOT))

# %%
import swisseph as swe
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
from tqdm import tqdm
from sklearn.metrics import classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
    AstroSettings,
    BodyPosition,
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

print("✓ All modules imported")

# %% [markdown]
# ## 2. House Calculation Functions

# %%
# Системы домов
HOUSE_SYSTEMS = {
    "P": "Placidus",
    "K": "Koch", 
    "O": "Porphyry",
    "R": "Regiomontanus",
    "C": "Campanus",
    "E": "Equal",
    "W": "Whole Sign",
}

DEFAULT_HOUSE_SYSTEM = b"P"  # Placidus - самая популярная

# Географические координаты для Bitcoin
# Используем координаты места где Сатоши возможно работал
# Или нейтральные (0, 0) для "глобального" актива
LATITUDE = 0.0   # Экватор - нейтральная позиция
LONGITUDE = 0.0  # Гринвич

# %%
def calculate_houses(dt: datetime, lat: float, lon: float, hsys: bytes = DEFAULT_HOUSE_SYSTEM) -> dict:
    """
    Вычисляет куспиды домов для заданного времени и места.
    
    Returns:
        dict с ключами:
        - cusps: list[float] - 12 куспидов домов (градусы зодиака)
        - asc: float - Асцендент
        - mc: float - MC (Medium Coeli)
        - armc: float - ARMC
        - vertex: float - Vertex
        - equasc: float - Equatorial Ascendant
    """
    # Конвертируем в Julian Day
    jd = swe.julday(dt.year, dt.month, dt.day, 
                    dt.hour + dt.minute/60.0 + dt.second/3600.0)
    
    # Рассчитываем дома
    cusps, ascmc = swe.houses(jd, lat, lon, hsys)
    
    return {
        "cusps": list(cusps),  # 12 куспидов (индекс 0 = 1-й дом)
        "asc": ascmc[0],       # Асцендент
        "mc": ascmc[1],        # MC
        "armc": ascmc[2],      # ARMC
        "vertex": ascmc[3],    # Vertex
        "equasc": ascmc[4],    # Equatorial Ascendant
    }

# %%
def get_house_positions(bodies: list, cusps: list) -> dict:
    """
    Определяет в каком доме находится каждая планета.
    
    Args:
        bodies: список BodyPosition
        cusps: список из 12 куспидов домов
        
    Returns:
        dict: {body_name: house_number (1-12)}
    """
    result = {}
    
    for body in bodies:
        lon = body.lon
        
        # Определяем дом
        for i in range(12):
            cusp_start = cusps[i]
            cusp_end = cusps[(i + 1) % 12]
            
            # Учитываем переход через 0° Овна
            if cusp_start > cusp_end:
                if lon >= cusp_start or lon < cusp_end:
                    result[body.body] = i + 1
                    break
            else:
                if cusp_start <= lon < cusp_end:
                    result[body.body] = i + 1
                    break
        else:
            result[body.body] = 1  # Fallback
            
    return result

# %%
def calculate_aspects_to_cusps(
    transit_bodies: list,
    natal_cusps: list,
    orb: float = 8.0,
) -> pd.DataFrame:
    """
    Вычисляет аспекты транзитных планет к куспидам натальных домов.
    
    Args:
        transit_bodies: список BodyPosition транзитных планет
        natal_cusps: список из 12 куспидов
        orb: орбис в градусах
        
    Returns:
        DataFrame с аспектами к куспидам
    """
    aspects = {
        "Conjunction": 0,
        "Sextile": 60,
        "Square": 90,
        "Trine": 120,
        "Opposition": 180,
    }
    
    cusp_names = [f"Cusp_{i+1}" for i in range(12)]
    cusp_names[0] = "ASC"  # 1-й дом = Асцендент
    cusp_names[9] = "MC"   # 10-й дом = MC
    
    results = []
    
    for body in transit_bodies:
        for i, cusp_lon in enumerate(natal_cusps):
            for asp_name, asp_angle in aspects.items():
                # Разница углов
                diff = abs(body.lon - cusp_lon)
                if diff > 180:
                    diff = 360 - diff
                    
                # Орбис
                orb_diff = abs(diff - asp_angle)
                if orb_diff <= orb:
                    results.append({
                        "transit_body": body.body,
                        "cusp": cusp_names[i],
                        "aspect": asp_name,
                        "orb": orb_diff,
                    })
    
    return pd.DataFrame(results)

# %%
def build_house_features(
    transit_bodies: list,
    natal_cusps: list,
    natal_houses: dict,
    orb: float = 8.0,
) -> dict:
    """
    Строит признаки на основе домов.
    
    Returns:
        dict с признаками для одной даты
    """
    features = {}
    
    # 1. Позиции транзитных планет в натальных домах
    transit_houses = get_house_positions(transit_bodies, natal_cusps)
    for body_name, house_num in transit_houses.items():
        # One-hot encoding домов
        for h in range(1, 13):
            features[f"transit_{body_name}_house_{h}"] = 1 if house_num == h else 0
    
    # 2. Аспекты к куспидам
    aspects = {
        "Conjunction": 0,
        "Sextile": 60,
        "Square": 90,
        "Trine": 120,
        "Opposition": 180,
    }
    
    cusp_names = ["ASC", "H2", "H3", "IC", "H5", "H6", "DSC", "H8", "H9", "MC", "H11", "H12"]
    
    for body in transit_bodies:
        for i, cusp_lon in enumerate(natal_cusps):
            for asp_name, asp_angle in aspects.items():
                diff = abs(body.lon - cusp_lon)
                if diff > 180:
                    diff = 360 - diff
                    
                orb_diff = abs(diff - asp_angle)
                # Бинарный признак: есть аспект или нет
                feat_name = f"transit_{body.body}_{asp_name}_{cusp_names[i]}"
                features[feat_name] = 1 if orb_diff <= orb else 0
    
    return features

# %% [markdown]
# ## 3. Configuration

# %%
# ══════════════════════════════════════════════════════════════════════════════════
# ⏰ НАСТРОЙКИ ПОИСКА ВРЕМЕНИ
# ══════════════════════════════════════════════════════════════════════════════════

# Лучшая дата из birthdate_search
BIRTH_DATE = date(2009, 10, 10)

# Шаг поиска времени
TIME_STEP_MINUTES = 5  # каждые 5 минут

# Диапазон времени (UTC)
TIME_START = datetime(2009, 10, 10, 0, 0, 0, tzinfo=timezone.utc)
TIME_END = datetime(2009, 10, 10, 23, 59, 59, tzinfo=timezone.utc)

# Географические координаты
# Хельсинки, Финляндия (Martti Malmi / New Liberty Standard)
# Место первой котировки BTC/USD
LATITUDE, LONGITUDE = 60.1699, 24.9384

# Система домов
HOUSE_SYSTEM = b"P"  # Placidus

# Орбис для аспектов к куспидам
CUSP_ORB = 5.0  # градусов

# Лучшие параметры из предыдущих поисков
BEST_ASTRO_CONFIG = {
    "coord_mode": "both",
    "orb_mult": 0.15,
    "gauss_window": 300,
    "gauss_std": 70.0,
    "exclude_bodies": None,
}

BEST_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
}

# Подсчёт вариантов
total_minutes = 24 * 60
total_variants = total_minutes // TIME_STEP_MINUTES

print(f"⏰ Birth date: {BIRTH_DATE}")
print(f"   Time range: {TIME_START.strftime('%H:%M')} - {TIME_END.strftime('%H:%M')} UTC")
print(f"   Step: {TIME_STEP_MINUTES} min")
print(f"   Total variants: {total_variants}")
print(f"   Location: lat={LATITUDE}, lon={LONGITUDE}")
print(f"   House system: {HOUSE_SYSTEMS.get(HOUSE_SYSTEM.decode(), 'Unknown')}")

# %% [markdown]
# ## 4. Load Data

# %%
# Load market data
df_market = load_market_data()
DATA_START = "2017-11-01"
df_market = df_market[df_market["date"] >= DATA_START].reset_index(drop=True)

print(f"Market data: {len(df_market)} rows")

# %%
# Create labels
df_labels = create_balanced_labels(
    df_market,
    gauss_window=BEST_ASTRO_CONFIG["gauss_window"],
    gauss_std=BEST_ASTRO_CONFIG["gauss_std"],
)
print(f"Labels: {len(df_labels)} rows")

# %%
# Initialize
settings = init_ephemeris()
_, device = check_cuda_available()
print(f"Device: {device}")

# %% [markdown]
# ## 5. Pre-compute Transit Bodies

# %%
print("📍 Pre-computing transit bodies...")

df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
    df_market["date"], settings,
    coord_mode=BEST_ASTRO_CONFIG["coord_mode"],
    progress=True
)
bodies_by_date = geo_by_date

print(f"✓ Transit bodies ready: {len(df_bodies)} rows")

# %%
print("\n🌙 Computing phases...")
df_phases = calculate_phases_for_dates(bodies_by_date, progress=True)
print(f"✓ Phases: {len(df_phases)} rows")

# %% [markdown]
# ## 6. Time Search Function

# %%
def evaluate_birth_time(
    birth_dt: datetime,
    df_market: pd.DataFrame,
    df_labels: pd.DataFrame,
    df_bodies: pd.DataFrame,
    bodies_by_date: dict,
    df_phases: pd.DataFrame,
    settings: AstroSettings,
    config: dict,
    xgb_params: dict,
    lat: float,
    lon: float,
    hsys: bytes,
    cusp_orb: float,
    device: str,
) -> dict:
    """
    Оценка одного времени рождения с учётом домов.
    """
    try:
        # 1. Рассчитываем дома для этого времени
        houses = calculate_houses(birth_dt, lat, lon, hsys)
        natal_cusps = houses["cusps"]
        
        # 2. Получаем натальные позиции планет
        birth_dt_str = birth_dt.strftime("%Y-%m-%dT%H:%M:%S")
        natal_bodies = get_natal_bodies(birth_dt_str, settings)
        
        # 3. Позиции натальных планет в домах
        natal_houses = get_house_positions(natal_bodies, natal_cusps)
        
        # 4. Строим признаки домов для каждой даты
        house_features_list = []
        
        for dt, transit_bodies in bodies_by_date.items():
            # Базовые признаки домов
            hf = build_house_features(
                transit_bodies, natal_cusps, natal_houses, orb=cusp_orb
            )
            hf["date"] = dt
            house_features_list.append(hf)
        
        df_house_features = pd.DataFrame(house_features_list)
        
        # 5. Вычисляем транзиты к натальным планетам
        df_transits = calculate_transits_for_dates(
            bodies_by_date, natal_bodies, settings,
            orb_mult=config["orb_mult"],
            progress=False,
        )
        
        # 6. Вычисляем аспекты между транзитными планетами
        df_aspects = calculate_aspects_for_dates(
            bodies_by_date, settings,
            orb_mult=config["orb_mult"],
            progress=False,
        )
        
        # 7. Строим полные признаки (без домов)
        df_features_base = build_full_features(
            df_bodies,
            df_aspects,
            df_transits=df_transits,
            df_phases=df_phases,
            include_pair_aspects=True,
            include_transit_aspects=True,
            exclude_bodies=config["exclude_bodies"],
        )
        
        # 8. Объединяем с признаками домов
        df_features_base["date"] = pd.to_datetime(df_features_base["date"])
        df_house_features["date"] = pd.to_datetime(df_house_features["date"])
        
        df_features = df_features_base.merge(df_house_features, on="date", how="left")
        df_features = df_features.fillna(0)
        
        # 9. Merge с метками
        df_dataset = merge_features_with_labels(df_features, df_labels)
        
        if len(df_dataset) < 100:
            return {"error": "Too few samples", "birth_time": birth_dt}
        
        # 10. Split
        train_df, val_df, test_df = split_dataset(df_dataset)
        feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
        
        X_train, y_train = prepare_xy(train_df, feature_cols)
        X_val, y_val = prepare_xy(val_df, feature_cols)
        X_test, y_test = prepare_xy(test_df, feature_cols)
        
        # 11. Train
        model = train_xgb_model(
            X_train, y_train, X_val, y_val,
            feature_cols, n_classes=2, device=device,
            **xgb_params
        )
        
        # 12. Tune threshold
        best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min")
        
        # 13. Predict
        y_pred = predict_with_threshold(model, X_test, threshold=best_t)
        
        # 14. Metrics
        report = classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["DOWN", "UP"], output_dict=True, zero_division=0
        )
        
        recall_down = report["DOWN"]["recall"]
        recall_up = report["UP"]["recall"]
        mcc = matthews_corrcoef(y_test, y_pred)
        
        return {
            "birth_time": birth_dt,
            "time_str": birth_dt.strftime("%H:%M"),
            "recall_down": recall_down,
            "recall_up": recall_up,
            "recall_min": min(recall_down, recall_up),
            "recall_gap": abs(recall_down - recall_up),
            "mcc": mcc,
            "threshold": best_t,
            "n_features": len(feature_cols),
            "asc": houses["asc"],
            "mc": houses["mc"],
        }
        
    except Exception as e:
        return {"error": str(e), "birth_time": birth_dt}

# %% [markdown]
# ## 7. Run Time Search

# %%
print("=" * 80)
print("⏰ BIRTH TIME SEARCH (WITH HOUSES)")
print("=" * 80)
print(f"   Date: {BIRTH_DATE}")
print(f"   Variants: {total_variants}")
print()

# %%
results = []
best_so_far = {"recall_min": 0, "birth_time": None}

current_time = TIME_START

with tqdm(total=total_variants, desc="Time search") as pbar:
    while current_time <= TIME_END:
        result = evaluate_birth_time(
            current_time,
            df_market, df_labels, df_bodies, bodies_by_date, df_phases,
            settings, BEST_ASTRO_CONFIG, BEST_XGB_PARAMS,
            LATITUDE, LONGITUDE, HOUSE_SYSTEM, CUSP_ORB, device
        )
        results.append(result)
        
        if "error" not in result:
            r_min = result["recall_min"]
            if r_min > best_so_far["recall_min"]:
                best_so_far = result.copy()
                tqdm.write(f"🏆 NEW BEST: {result['time_str']} → R_MIN={r_min:.3f} MCC={result['mcc']:.3f} ASC={result['asc']:.1f}°")
        else:
            if best_so_far["birth_time"] is None: # Only print first few errors
                 tqdm.write(f"❌ Error for {result.get('time_str', 'unknown')}: {result['error']}")
        
        current_time += timedelta(minutes=TIME_STEP_MINUTES)
        pbar.update(1)

# %%
# Results
print("\n" + "=" * 80)
print("📊 TIME SEARCH RESULTS")
print("=" * 80)

df_results = pd.DataFrame([r for r in results if "error" not in r])

if len(df_results) > 0:
    df_results = df_results.sort_values("recall_min", ascending=False)
    
    print("\n🏆 TOP 20 BEST TIMES:")
    print("-" * 80)
    print(f"{'#':<3} {'Time':<8} {'R_MIN':<8} {'R_UP':<8} {'R_DOWN':<8} {'MCC':<8} {'ASC':<8} {'MC':<8}")
    print("-" * 80)
    
    for idx, (i, row) in enumerate(df_results.head(20).iterrows()):
        print(f"{idx+1:<3} {row['time_str']:<8} {row['recall_min']:.3f}    {row['recall_up']:.3f}    "
              f"{row['recall_down']:.3f}    {row['mcc']:.3f}    {row['asc']:.1f}°    {row['mc']:.1f}°")
    
    # Save
    results_path = PROJECT_ROOT / "data" / "market" / "reports" / "birthtime_search_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n💾 Results saved: {results_path}")
    
    best = df_results.iloc[0]
    print(f"\n⏰ BEST BIRTH TIME: {BIRTH_DATE} {best['time_str']} UTC")
    print(f"   R_MIN = {best['recall_min']:.3f}")
    print(f"   MCC   = {best['mcc']:.3f}")
    print(f"   ASC   = {best['asc']:.1f}°")
    print(f"   MC    = {best['mc']:.1f}°")

# %% [markdown]
# ## 8. Visualization

# %%
if len(df_results) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort by time for plotting
    df_plot = df_results.sort_values("birth_time")
    hours = [dt.hour + dt.minute/60 for dt in df_plot["birth_time"]]
    
    # R_MIN by time
    ax1 = axes[0, 0]
    ax1.plot(hours, df_plot["recall_min"], 'b-', alpha=0.7)
    ax1.scatter([best["birth_time"].hour + best["birth_time"].minute/60], 
                [best["recall_min"]], c='red', s=100, zorder=5)
    ax1.set_xlabel("Hour (UTC)")
    ax1.set_ylabel("R_MIN")
    ax1.set_title("R_MIN by Birth Time")
    ax1.grid(True, alpha=0.3)
    
    # MCC by time
    ax2 = axes[0, 1]
    ax2.plot(hours, df_plot["mcc"], 'orange', alpha=0.7)
    ax2.scatter([best["birth_time"].hour + best["birth_time"].minute/60], 
                [best["mcc"]], c='red', s=100, zorder=5)
    ax2.set_xlabel("Hour (UTC)")
    ax2.set_ylabel("MCC")
    ax2.set_title("MCC by Birth Time")
    ax2.grid(True, alpha=0.3)
    
    # ASC distribution
    ax3 = axes[1, 0]
    ax3.scatter(df_plot["asc"], df_plot["recall_min"], alpha=0.5)
    ax3.scatter([best["asc"]], [best["recall_min"]], c='red', s=100, zorder=5)
    ax3.set_xlabel("Ascendant (degrees)")
    ax3.set_ylabel("R_MIN")
    ax3.set_title("R_MIN by Ascendant")
    ax3.grid(True, alpha=0.3)
    
    # MC distribution
    ax4 = axes[1, 1]
    ax4.scatter(df_plot["mc"], df_plot["recall_min"], alpha=0.5)
    ax4.scatter([best["mc"]], [best["recall_min"]], c='red', s=100, zorder=5)
    ax4.set_xlabel("MC (degrees)")
    ax4.set_ylabel("R_MIN")
    ax4.set_title("R_MIN by MC")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
print("\n" + "=" * 80)
print("✅ BIRTH TIME SEARCH COMPLETE!")
print("=" * 80)
