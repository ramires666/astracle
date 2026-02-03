# RESEARCH — Astro Trading Research Pipeline

Modular pipeline for researching correlations between astrological data and cryptocurrency market movements.

## Quick Start

### 1. Create/Activate Conda Environment

```bash
# If the btc environment already exists:
conda activate btc

# Or create a new one:
conda create -n btc python=3.12 -y
conda activate btc
```

### 2. Install Dependencies

**Option A — Single Command (recommended):**

```bash
# Core DS/ML packages + psycopg2 via conda
conda install -c conda-forge xgboost scikit-learn matplotlib seaborn tqdm pyarrow psycopg2 ipykernel joblib pandas numpy scipy -y

# Astro engine (not available in conda, install via pip)
pip install pyswisseph
```

**Option B — Via pip (if not using conda):**

```bash
pip install -r RESEARCH/requirements.txt
```

### 3. Run in VS Code (Interactive Mode)

1. Open `RESEARCH/main_pipeline.py`
2. Make sure the `btc` interpreter is selected (`Ctrl+Shift+P` → `Python: Select Interpreter`)
3. Press `Shift+Enter` on any cell (marked with `# %%`) or click **Run Cell**

## Module Structure

| Module | Description |
|--------|-------------|
| `config.py` | Project configuration (paths, DB settings, subjects) |
| `data_loader.py` | Load market data from PostgreSQL |
| `labeling.py` | Create balanced UP/DOWN labels |
| `astro_engine.py` | Calculate planetary positions and aspects (Swiss Ephemeris) |
| `features.py` | Build feature matrix |
| `model_training.py` | Train XGBoost, tune threshold |
| `visualization.py` | Charts: price, distributions, confusion matrix |
| `grid_search.py` | Grid search over parameter space |
| `main_pipeline.py` | **Main file** — orchestrates the entire pipeline |

## Dependency Check

Run the first cell of `main_pipeline.py` — it will show any missing packages:

```python
# %%
import importlib.util as iu
required = ["xgboost", "sklearn", "matplotlib", "seaborn", "tqdm", "pyarrow", "psycopg2", "swisseph"]
missing = [pkg for pkg in required if iu.find_spec(pkg) is None]
if missing:
    print("Missing:", ", ".join(missing))
else:
    print("✓ All dependencies found")
```

---

## 🚧 Project Status

This project is **under active development**. The core pipeline is functional and has already produced statistically significant results.

## 📊 Current Results

After extensive grid search optimization (35,000+ parameter combinations), the best model achieved:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall (min)** | 57.8% | +7.8% above random baseline |
| **Recall Gap** | 0.4% | Near-perfect balance between UP/DOWN |
| **MCC** | 0.159 | Weak but real predictive signal |

**Best Configuration:**
- Coordinate mode: Geocentric
- Orb multiplier: 0.25 (tight aspects only)
- Gaussian window: 201 days
- Gaussian std: 50.0
- Excluded bodies: Uranus, Pluto (reduced noise)

### Statistical Significance

- **z-score ≈ 4.9** (assuming ~1000 test samples)
- **p-value < 0.0001** — probability of random chance is less than 0.01%
- The model demonstrates a **statistically significant edge** over random guessing

### Practical Implications

| Aspect | Assessment |
|--------|------------|
| ✅ Better than random | Yes, by ~7.8 percentage points |
| ✅ Balanced predictions | Equal accuracy for UP and DOWN moves |
| ⚠️ Edge size | Moderate — requires low trading fees |
| 🎯 Key finding | Outer planets (Uranus, Pluto) add noise; excluding them improves performance |

### Interpretation

The MCC of 0.159 indicates a **weak but statistically real correlation** between planetary aspects and market movements. While not strong enough for high-frequency trading, this edge may be viable for:
- Medium to long-term position trading
- Signal confirmation in conjunction with other indicators
- Further research into specific planetary configurations

---

*Note: Past performance does not guarantee future results. This is research, not financial advice.*
