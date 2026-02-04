# 🔬 Astro Trading Model Research Timeline

## Overview

This document reconstructs the research and development process of the astro-trading XGBoost model, tracing the journey from the initial commit through achieving **60.3% recall** (minimum recall across classes).

> **📦 Reproducibility Package:** See [`RESEARCH-REPRODUCE/`](../RESEARCH-REPRODUCE/) folder with step-by-step scripts to reproduce the 60.3% result.

---

## 📊 METRIC PROGRESSION SUMMARY

| Phase | R_MIN | MCC | Key Discovery | Features |
|-------|-------|-----|---------------|----------|
| Random baseline | 50.0% | 0.0 | - | - |
| Initial XGB | 50.4% | 0.062 | Grid search started | ~300 |
| +Body exclusion | 55.7% | 0.124 | Uranus+Pluto = noise | ~500 |
| Optimized params | 57.0% | 0.173 | Tight orbs work | ~600 |
| Best grid search | 57.9% | 0.164 | orb=0.05, win=150 | ~650 |
| +Natal transits | 59.0% | 0.195 | orb=0.075, win=200 | ~1200 |
| **Deep tuning** | **60.3%** | **0.315** | birth=2009-10-10 | **2040** |

---

## 📅 Complete Commit Timeline

| # | Commit | Date | Message | Key Changes |
|---|--------|------|---------|-------------|
| 1 | `1a9bc0d` | 2026-01-29 | "first commit after 2 days of prototyping" | Initial project |
| 2 | `bbee0bc` | 2026-02-02 | "..." | **RESEARCH module** |
| 3 | `849d623` | 2026-02-02 | "planet phases, optimized grid search" | +Phases, +README |
| 4 | `5d332e7` | 2026-02-02 | "pre refactoring" | Cleanup |
| 5 | `e589705` | 2026-02-02 | "pre refactor speedup" | +numba_utils |
| 6 | `d49215d` | 2026-02-02 | "grid search" | +grid_search_pkg |
| 7-9 | ... | 2026-02-03 | Documentation | README updates |
| 10 | `fa898d2` | 2026-02-03 | "finetuning search" | +single_body_search |
| 11-17 | ... | 2026-02-03 | XGB tuning | **R_MIN=0.587** |
| 18 | `f4a584b` | 2026-02-03 | "full grid search + birthday search" | +birthdate_search |
| 19 | `e5037b5` | 2026-02-03 | "grid search + production" | +birthdate_deep_search |
| 20-22 | ... | 2026-02-03 | Finalization | **R_MIN=0.603** |
| 23-28 | ... | 2026-02-03 | Body ablation | +body_ablation |
| 29+ | ... | 2026-02-03→04 | Ternary | +ternary classification |

---

## 🛠️ Phase 0: Pre-Git Prototyping

**Duration:** ~2 days before first commit

### Initial Files (in `notebooks/`):
- `astro_xgboost_training.ipynb`
- `astro_xgboost_training_balanced.ipynb`

### Baseline Metrics:
- **Majority baseline:** acc=0.50, bal_acc=0.50
- **Prev-label baseline:** acc=0.48, bal_acc=0.48

---

## 🏗️ Phase 1: RESEARCH Module Creation

### Commit `bbee0bc` (2026-02-02)

**Created Files:**

| File | Lines | Purpose |
|------|-------|---------|
| `main_pipeline.py` | 357 | Central orchestrator |
| `config.py` | ~100 | Configuration |
| `data_loader.py` | ~80 | PostgreSQL loading |
| `labeling.py` | ~100 | Balanced labels |
| `astro_engine.py` | 230 | Swiss Ephemeris |
| `features.py` | 149 | Feature matrix |
| `model_training.py` | ~200 | XGBoost training |
| `visualization.py` | ~200 | Charts |
| `grid_search.py` | ~200 | Param search |

### Initial Configuration:
```python
# labeling.py
gauss_window = 201
gauss_std = 50.0
horizon = 1
move_share = 0.5
label_mode = "balanced_detrended"

# features.py TODOs:
# TODO: Grid search for excluding individual astro objects
# TODO: Add moon phases and other planet phases
# TODO: Add houses for birth date grid search
```

---

## 📊 Phase 2: Feature Expansion

### Commit `849d623` (2026-02-02) - "planet phases"

**Major Additions:**
- `README.md` - Project documentation (Russian)
- `main_pipeline.ipynb` - Jupyter notebook version
- **Planet Phases** - Moon illumination, lunar day, elongations

**astro_engine.py expansion:** 230 → 977 lines (+747)

**New features in `features.py`:**
```python
# ФАЗЫ И ЭЛОНГАЦИИ (df_phases, НОВОЕ!):
# - moon_phase_angle : угол фазы Луны (0-360°)
# - moon_phase_ratio : нормализованная фаза (0=новолуние, 0.5=полнолуние)
# - moon_illumination : освещённость Луны (0-1)
# - lunar_day : лунный день (1-29.5)
# - Mercury_elongation : элонгация Меркурия от Солнца
# - Venus_elongation : элонгация Венеры
```

---

## 🔍 Phase 3: Grid Search Implementation

### Commit `d49215d` (2026-02-02)

**Search Space:**
```python
class GridSearchConfig:
    orb_multipliers = [0.8, 1.0, 1.2]
    gauss_windows = [101, 151, 201]
    gauss_stds = [30.0, 50.0, 70.0]
    coord_modes = ["geo", "helio", "both"]
    max_exclude = 0  # body ablation
```

---

## 📈 Phase 4: Grid Search Results

### First Reports (2026-02-03):

**File:** `grid_search_20260203_142502.csv` (3.7KB, ~20 rows)

| coord_mode | gauss_window | orb_mult | exclude_bodies | R_MIN | MCC |
|------------|--------------|----------|----------------|-------|-----|
| geo | 75 | 0.05 | Mercury | 0.495 | 0.056 |
| geo | 75 | 0.05 | MeanNode,Pluto | 0.491 | 0.045 |
| geo | 75 | 0.1 | none | 0.478 | 0.010 |
| geo | 75 | 0.05 | Pluto,Saturn | 0.439 | 0.043 |

**Finding:** Early stages showed ~49-50% R_MIN (barely above random).

---

**File:** `grid_search_20260203_152247.csv` (444KB, ~1400 rows)

Top result after expanded search:

| coord_mode | window | std | orb | exclude | R_MIN | MCC |
|------------|--------|-----|-----|---------|-------|-----|
| both | 75 | 90.0 | 0.25 | Pluto,Venus | **0.525** | 0.057 |

---

## 🎯 Phase 5: Hyperparameter Tuning

### Commit `fa898d2` - "finetuning search for perfection"

**Created:**
- `single_body_search.ipynb` - Which bodies matter
- `xgb_hyperparam_search.py` - XGB param search

### XGB Grid (from `xgb_hyperparam_search.ipynb`):
```python
{
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
```

### Metric Progression (from notebook output):
```
🚀 NEW BEST: R_MIN=0.5044 MCC=0.0620 Gap=0.0531 | Orb=0.05 Win=150 Excl=None
🚀 NEW BEST: R_MIN=0.5575 MCC=0.1239 Gap=0.0088 | Orb=0.05 Win=150 Excl=['Uranus', 'Pluto']
🚀 NEW BEST: R_MIN=0.5708 MCC=0.1726 Gap=0.0310 | Orb=0.05 Win=150 Excl=['Uranus', 'Pluto']
🚀 NEW BEST: R_MIN=0.5796 MCC=0.1637 Gap=0.0044 | Orb=0.05 Win=150 Excl=['Uranus', 'Pluto']
🚀 NEW BEST: R_MIN=0.5903 MCC=0.1948 Gap=0.0141 | Orb=0.075 Win=200 Excl=None
```

### Established Baseline:
```
baseline: R_MIN=0.587, MCC=0.182
```

### Winner Configuration at This Point:
```python
{
    'coord_mode': 'both',
    'orb_mult': 0.075,
    'gauss_window': 200,
    'gauss_std': 70.0,
    'exclude_bodies': None,
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.03,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'R_MIN': 0.5903,
    'MCC': 0.195
}
```

---

## 🌟 Phase 6: Birth Date Search

### Commit `f4a584b` - "full grid search + birthday search"

**New Hypothesis:**
> "Bitcoin has a 'birth chart'. Adding transits to natal chart might improve predictions."

**Created:** `birthdate_search.ipynb`

### Known Birth Date Candidates:
| Date | Event | Initial R_MIN |
|------|-------|---------------|
| 2008-08-18 | Domain registration | 0.587 |
| 2008-10-31 | Whitepaper published | 0.343 |
| 2009-01-03 | Genesis block | 0.565 |
| 2009-01-09 | First transaction | 0.548 |
| 2009-10-10 | First exchange rate | 0.578 |

### Full Year Search (2009-01-01 → 2009-12-31):
```
🏆 NEW BEST: 2009-01-01 → R_MIN=0.570 MCC=0.142
🏆 NEW BEST: 2009-03-21 → R_MIN=0.574 MCC=0.164
🏆 NEW BEST: 2009-06-14 → R_MIN=0.577 MCC=0.181
🏆 NEW BEST: 2009-10-10 → R_MIN=0.578 MCC=0.205
🏆 NEW BEST: 2009-12-03 → R_MIN=0.581 MCC=0.172
```

### Top 20 Best Birth Dates:
```
#   Date         R_MIN    R_UP     R_DOWN   MCC
1   2009-12-03   0.581    0.581    0.593    0.172
2   2009-10-10   0.578    0.626    0.578    0.205  ← Best MCC!
3   2009-10-12   0.578    0.581    0.578    0.159
10  2009-01-18   0.570    0.581    0.570    0.151
11  2009-01-01   0.570    0.572    0.570    0.142
```

**Initial Winner:** 2009-12-03 (R_MIN=0.581)  
**Best MCC:** 2009-10-10 (MCC=0.205) ← This date was later used!

---

## 🏆 Phase 7: Deep Model Tuning → 60.3%

### Commit `e5037b5` - "grid search + started production"

**Created:**
- `birthdate_deep_search.py`
- `birthdate_deep_search.ipynb` ← **THE 60.3% FILE**

### Configuration in `birthdate_deep_search.py`:
```python
TARGET_DATE = date(2009, 10, 10)  # Economic Birth

ASTRO_CONFIG = {
    "coord_mode": "both",      # Geo + Helio
    "orb_mult": 0.15,          # Slightly wider than 0.1
    "gauss_window": 300,       # Wider window
    "gauss_std": 70.0,         # Higher std
    "exclude_bodies": None,    # All bodies included
}

PARAM_GRID = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [3, 4, 6, 8, 10],
    "learning_rate": [0.01, 0.03],
    "colsample_bytree": [0.6, 0.8],
    "subsample": [0.8],
}
```

### Features Composition:
```
Total: 2040 features
├── Body positions (geo + helio): ~400
├── Aspect pairs: ~600
├── Transit-to-natal aspects: ~900
└── Phases + elongations: ~140
```

### Grid Search Output (36 combinations):
```
🏆 TOP 10 MODELS:
    n_estimators  max_depth  learning_rate  colsample_bytree  R_MIN    MCC
2            500          6           0.03               0.6  0.6029  0.315
14           900          6           0.03               0.6  0.6029  0.315
26          1300          6           0.03               0.6  0.6029  0.315
0            500          6           0.05               0.6  0.5970  0.309
```

### 🥇 WINNER PARAMS:
```python
{
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.03,
    'colsample_bytree': 0.6,
    'subsample': 0.8,
    'R_MIN': 0.6029,  # 60.3%
    'MCC': 0.315
}
```

### SUCCESS MESSAGE:
```
🚀 SUCCESS! Deep model beat baseline! (0.603 > 0.587)

Model exported to: ../models_artifacts/btc_astro_predictor.joblib
   Features: 2040
   R_MIN: 0.603
   MCC: 0.315
```

---

## 🔑 Key Research Findings

### 1. Birth Date = 2009-10-10 (Economic Birth)
First published exchange rate. Not genesis block (2009-01-03), not domain registration (2008-08-18).

### 2. Tight Orbs Win
- `orb_mult = 0.1-0.15` (default is 1.0)
- Only exact planetary aspects matter

### 3. Both Coordinate Systems
- `coord_mode = "both"` (geo + helio)
- 2x more body positions

### 4. Moderate XGB Depth
- `max_depth = 6` optimal
- Not too deep (overfitting), not too shallow

### 5. Feature Explosion with Transits
- From ~700 → 2040 features
- Transit-to-natal aspects added ~900 new features

### 6. MCC Improvement
- From 0.182 → 0.315 (73% increase!)
- Much stronger predictive signal

---

## 📁 Key Files Summary

| File | Purpose | Key Output |
|------|---------|------------|
| `main_pipeline.py` | Core pipeline | - |
| `grid_search.py` | Param search | CSV reports |
| `single_body_search.ipynb` | Body ablation | Uranus/Pluto = noise |
| `xgb_hyperparam_search.ipynb` | XGB tuning | R_MIN=0.587 baseline |
| `birthdate_search.ipynb` | Birth date search | 2009-10-10 best MCC |
| **`birthdate_deep_search.ipynb`** | **Deep tuning** | **R_MIN=0.603** |

---

## 📋 TODO: Files to Deep-Analyze

- [ ] `Pre-research/astro_xgboost_training_balanced.ipynb` - Initial baseline
- [ ] `body_ablation_research.ipynb` - Which bodies affect results
- [ ] `birthtime_search.ipynb` - Time-level precision
- [ ] `xgb_tuning_top30.ipynb` - Feature selection

---

## 🔮 Post-60.3% Development

After the 60.3% milestone:

1. **Ternary Classification** - 3 classes (DOWN/SIDEWAYS/UP)
2. **Sideways Penalty** - Better handling of sideways markets
3. **Body Ablation for Ternary** - Ongoing optimization

---

*Document based on git history analysis and notebook output extraction.*
*Last metric found: R_MIN=0.603 in `birthdate_deep_search.ipynb`*
