# 🔬 RESEARCH REPRODUCIBILITY PACKAGE

## Goal: Reproduce the 60.3% Recall Model

This package contains all necessary files to reproduce the research journey 
from baseline (~50%) to 60.3% recall step by step.

---

## ⚠️ IMPORTANT: Prerequisites

### 1. Database
You need PostgreSQL with BTC market data:
```bash
# Check database connection
psql -U postgres -d btc_db -c "SELECT COUNT(*) FROM ohlcv_daily WHERE symbol='BTCUSD';"
```

### 2. Conda Environment
```bash
conda activate btc
# Or create new:
conda create -n btc python=3.12 -y
conda activate btc
conda install -c conda-forge xgboost scikit-learn matplotlib seaborn tqdm pyarrow psycopg2 ipykernel joblib pandas numpy scipy -y
pip install pyswisseph
```

### 3. Swiss Ephemeris Files
Ensure you have ephemeris files at the configured path.

---

## 📁 Package Structure

```
RESEARCH-REPRODUCE/
├── README.md                    # THIS FILE - Instructions
├── configs/                     # Astro configurations (bodies, aspects)
├── src/                         # Core source modules
├── step1_baseline/              # Initial pipeline (R_MIN ~50%)
├── step2_grid_search/           # Grid search implementation  
├── step3_body_ablation/         # Body exclusion experiments
├── step4_birthdate_search/      # Bitcoin birth date search
├── step5_deep_tuning/           # Final 60.3% achievement
└── reproduce_all.py             # Master reproduction script
```

---

## 🚀 REPRODUCTION STEPS

### STEP 1: Run Baseline Pipeline (Expected: R_MIN ~50%)

**Purpose:** Establish baseline with default parameters.

**Files:** `step1_baseline/`
- `main_pipeline.py` - Core pipeline
- `config.py`, `data_loader.py`, `labeling.py`
- `astro_engine.py`, `features.py`, `model_training.py`

**Configuration at this stage:**
```python
gauss_window = 201
gauss_std = 50.0
orb_mult = 1.0   # DEFAULT - not optimized yet
coord_mode = "geo"
exclude_bodies = None
```

**Run:**
```bash
cd step1_baseline
python main_pipeline.py
```

**Expected Result:**
```
R_MIN: ~0.50 (around random baseline)
MCC: ~0.0
```

---

### STEP 2: Grid Search for Parameters (Expected: R_MIN ~52-55%)

**Purpose:** Find optimal Gaussian and orb parameters.

**Files:** `step2_grid_search/grid_search.py`

**Key Parameters Being Searched:**
```python
orb_multipliers = [0.8, 1.0, 1.2, 0.5, 0.25, 0.1, 0.05]
gauss_windows = [75, 101, 151, 201, 300]
gauss_stds = [30.0, 50.0, 70.0, 90.0]
coord_modes = ["geo", "helio", "both"]
```

**Run:**
```bash
cd step2_grid_search
python grid_search.py
```

**Expected Result:**
```
Best so far: R_MIN ~0.52-0.55
Discovery: Tight orbs (0.05-0.1) work better!
```

---

### STEP 3: Body Ablation Study (Expected: R_MIN ~55-58%)

**Purpose:** Discover which celestial bodies add noise.

**Files:** `step3_body_ablation/xgb_hyperparam_search.py`

**Key Discovery:**
- Excluding Uranus and Pluto improves performance!
- These outer planets add noise to predictions.

**Configuration discovered:**
```python
exclude_bodies = ['Uranus', 'Pluto']
orb_mult = 0.05
gauss_window = 150
```

**Run:**
```bash
cd step3_body_ablation
python xgb_hyperparam_search.py
```

**Expected Results Progression:**
```
🚀 NEW BEST: R_MIN=0.5044 MCC=0.0620 | Orb=0.05 Excl=None
🚀 NEW BEST: R_MIN=0.5575 MCC=0.1239 | Orb=0.05 Excl=['Uranus', 'Pluto']
🚀 NEW BEST: R_MIN=0.5708 MCC=0.1726 | Orb=0.05 Excl=['Uranus', 'Pluto']
🚀 NEW BEST: R_MIN=0.5796 MCC=0.1637 | Orb=0.05 Excl=['Uranus', 'Pluto']
```

**Baseline Established:**
```
baseline: R_MIN=0.587, MCC=0.182
```

---

### STEP 4: Bitcoin Birth Date Search (Expected: R_MIN ~57-58%)

**Purpose:** Find the best "birth date" for Bitcoin's natal chart.

**Hypothesis:** Bitcoin has a "birth chart". Adding transits to this natal chart 
might improve predictions.

**Files:** `step4_birthdate_search/birthdate_search.py`

**Candidate Dates:**
| Date | Event | Result |
|------|-------|--------|
| 2008-08-18 | Domain registration | R_MIN=0.587 |
| 2008-10-31 | Whitepaper published | R_MIN=0.343 |
| 2009-01-03 | Genesis block | R_MIN=0.565 |
| 2009-10-10 | First exchange rate | R_MIN=0.578, **MCC=0.205** ← BEST MCC |
| 2009-12-03 | ? | R_MIN=0.581 |

**Run:**
```bash
cd step4_birthdate_search
python birthdate_search.py
```

**Key Discovery:**
```
🏆 Best birth date by MCC: 2009-10-10 (Economic birth)
```

---

### STEP 5: Deep Model Tuning → 60.3% 🎯

**Purpose:** Deep XGBoost tuning with natal transits.

**Files:** `step5_deep_tuning/birthdate_deep_search.py`

**THE WINNING CONFIGURATION:**
```python
TARGET_DATE = date(2009, 10, 10)  # Economic Birth

ASTRO_CONFIG = {
    "coord_mode": "both",      # Geo + Helio = 2x features
    "orb_mult": 0.15,          # Tight aspects
    "gauss_window": 300,       # Wider window
    "gauss_std": 70.0,
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

**Run:**
```bash
cd step5_deep_tuning
python birthdate_deep_search.py
```

**Expected Output:**
```
🏆 TOP 10 MODELS:
    n_estimators  max_depth  learning_rate  colsample_bytree  R_MIN    MCC
2            500          6           0.03               0.6  0.6029  0.315

🥇 WINNER PARAMS:
{'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 
 'colsample_bytree': 0.6, 'subsample': 0.8, 
 'R_MIN': 0.6029, 'MCC': 0.315}

🚀 SUCCESS! Deep model beat baseline! (0.603 > 0.587)

Model exported to: ../models_artifacts/btc_astro_predictor.joblib
   Features: 2040
   R_MIN: 0.603
   MCC: 0.315
```

---

## 📊 Expected Metric Progression

| Step | Script | R_MIN | MCC | Key Change |
|------|--------|-------|-----|------------|
| 1 | main_pipeline.py | ~50% | ~0.0 | Baseline |
| 2 | grid_search.py | ~52% | ~0.04 | Param optimization |
| 3 | xgb_hyperparam_search.py | ~57-58% | ~0.17 | Body ablation |
| 4 | birthdate_search.py | ~58% | ~0.20 | Birth date found |
| 5 | birthdate_deep_search.py | **60.3%** | **0.315** | Deep tuning |

---

## 🔑 Key Discoveries Summary

1. **Tight Orbs Work:** `orb_mult = 0.1-0.15` (not default 1.0)
2. **Both Coordinates:** `coord_mode = "both"` doubles features
3. **Birth Date Matters:** 2009-10-10 (not genesis block!)
4. **Moderate Depth:** `max_depth = 6` optimal for XGBoost
5. **Natal Transits:** Add ~900 new features (transit-to-natal aspects)

---

## ⚠️ Troubleshooting

### If results don't match:

1. **Check data range:** Should be `2017-11-01` onwards
2. **Check random seed:** Should be consistent (42)
3. **Check train/test split:** 80/20 temporal split
4. **Check feature count:** Step 5 should have ~2040 features

### Common Issues:

```python
# Missing natal bodies
# Solution: Ensure Swiss Ephemeris files are available

# Database connection error
# Solution: Check PostgreSQL is running and credentials are correct

# Different results each run
# Solution: Set random_state=42 consistently
```

---

## 📋 Verification Checklist

- [ ] Step 1: R_MIN ≈ 0.50 (baseline)
- [ ] Step 2: R_MIN ≈ 0.52-0.55 (grid search improvement)
- [ ] Step 3: R_MIN ≈ 0.57-0.58 (body ablation + XGB tuning)
- [ ] Step 4: Best birth date = 2009-10-10 (MCC=0.205)
- [ ] Step 5: R_MIN = 0.603, MCC = 0.315 (FINAL TARGET)

---

*Original research conducted: 2026-02-02 → 2026-02-03*
*Reproducibility package created: 2026-02-04*
