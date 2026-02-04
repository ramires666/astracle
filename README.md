# 🔮 Bitcoin Astro Predictor

> **AI-powered Bitcoin price direction predictions using astrological analysis and machine learning.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 📖 Table of Contents

1. [What Is This?](#-what-is-this)
2. [How It Works](#-how-it-works)
3. [Quick Start](#-quick-start)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Running the Prediction Service](#-running-the-prediction-service)
7. [API Reference](#-api-reference)
8. [Training Your Own Model](#-training-your-own-model)
9. [Research Timeline](#-research-timeline)
10. [Reproducibility](#-reproducibility)
11. [Configuration](#-configuration)
12. [FAQ](#-frequently-asked-questions)
13. [Disclaimer](#-disclaimer)

---

## 🌟 What Is This?

This project predicts whether **Bitcoin's price will go UP or DOWN** tomorrow using a unique approach: **astrological chart analysis combined with machine learning**.

### The Core Idea

Just like humans have birth charts (horoscopes), Bitcoin has one too! We calculate Bitcoin's "natal chart" based on its economic birth date (October 10, 2009 - when the first BTC/USD exchange rate was established), and then analyze how current planetary positions (transits) interact with that chart.

### What You Get

- 📈 **90-day price direction forecasts** (UP/DOWN predictions for each day)
- 🎯 **60.3% accuracy** on the validation set (R_MIN metric)
- 📊 **Beautiful web dashboard** with interactive charts
- 🐳 **Docker-ready** for easy deployment
- 🔄 **Daily data updates** from your database

---

## 🧠 How It Works

### Step 1: Calculate Bitcoin's Natal Chart

Every celestial body (Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto) has a position in the sky. On October 10, 2009, we record where each planet was - this is Bitcoin's "birth chart."

### Step 2: Calculate Daily Transits

For each day we want to predict, we calculate where the planets are now and how they relate to Bitcoin's natal positions. These relationships are called **aspects**:

| Aspect | Angle | Meaning |
|--------|-------|---------|
| Conjunction | 0° | Planets are together (intense energy) |
| Sextile | 60° | Harmonious, opportunity |
| Square | 90° | Tension, challenge |
| Trine | 120° | Flow, ease |
| Opposition | 180° | Polarity, balance needed |

### Step 3: Build Features

We convert all this astronomical data into numbers (features) that the machine learning model can understand:

- Planet positions (longitude in degrees)
- Aspect strengths (gaussian-weighted based on orb)
- Transit aspects (current planets to natal planets)
- Moon phases and elongations

### Step 4: Train the Model

We use **XGBoost** (a powerful gradient boosting algorithm) to learn patterns between these astrological features and actual Bitcoin price movements.

### Step 5: Predict the Future

Given today's planetary positions, the model predicts tomorrow's price direction with ~60% accuracy.

---

## 🏗️ Dual Model Architecture

This project uses a **dual-model strategy** to ensure both **honest backtesting** and **optimal forecasting**:

### The Problem

When a machine learning model is trained on historical data and then tested on the same data, it shows artificially high accuracy (overfitting). To provide honest accuracy metrics while still using all available data for the best predictions, we use two separate models:

### Model 1: Split Model (for Backtest)

| Property | Value |
|----------|-------|
| **Purpose** | Show honest historical accuracy |
| **Training Data** | Train/Val/Test split (70/15/15) |
| **File** | `models_artifacts/btc_astro_predictor.joblib` |
| **Used For** | Backtest predictions (past dates) |

The split model is trained on only ~70% of historical data, leaving 30% as a true holdout. When we show "Historical Accuracy" on the UI, these are real out-of-sample predictions the model never saw during training.

### Model 2: Full Model (for Forecast)

| Property | Value |
|----------|-------|
| **Purpose** | Best possible future predictions |
| **Training Data** | ALL available data (2017-present) |
| **File** | `models_artifacts/btc_astro_predictor.full.joblib` |
| **Used For** | Future predictions (forecast) |

The full model is trained on 100% of available historical data. For predicting the actual future (which the model has never seen), using all available information gives the best results.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    generate_cache.py                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐        ┌─────────────────────┐         │
│  │   BACKTEST CACHE    │        │   FORECAST CACHE    │         │
│  │   (Past 3 years)    │        │   (Next 365 days)   │         │
│  └─────────┬───────────┘        └──────────┬──────────┘         │
│            │                               │                     │
│            ▼                               ▼                     │
│  ┌─────────────────────┐        ┌─────────────────────┐         │
│  │   SPLIT MODEL       │        │   FULL MODEL        │         │
│  │   (Honest Accuracy) │        │   (Best Forecast)   │         │
│  │   ~60% R_MIN        │        │   Trained on ALL    │         │
│  └─────────────────────┘        └─────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Daily Retraining (Docker)

In production, the system automatically retrains the FULL model daily when new price data arrives:

```bash
# Cron job example (runs at 6 AM UTC)
0 6 * * * python -m production_dev.daily_retrain
```

The `daily_retrain.py` script:
1. **Updates price data** from the database
2. **Retrains the FULL model** on all available historical data
3. **Regenerates the cache** using both models (split for backtest, full for forecast)

### Key Files

| File | Description |
|------|-------------|
| `train_full_model.py` | Trains the FULL model on all data |
| `generate_cache.py` | Generates prediction cache using dual models |
| `daily_retrain.py` | Daily automation script for Docker |
| `cache_service.py` | Memory cache for fast API responses |

### Why This Matters

- **Honest Metrics**: The Historical Accuracy badge shows real out-of-sample performance (~60%)
- **Best Predictions**: Future forecasts use all available information
- **Production Ready**: Daily retraining keeps the model updated with latest data
- **Transparent**: Users can trust the accuracy numbers shown on the dashboard



## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```powershell
# From Windows PowerShell (project is in WSL)
cd \\wsl$\Ubuntu-24.04\home\rut\ostrofun

# Build and run
docker-compose -f production_dev/docker-compose.yml up --build

# Open in browser
start http://localhost:9742
```

### Option 2: Run Locally (Development)

```bash
# From WSL terminal
cd /home/rut/ostrofun

# Install dependencies
pip install -r production_dev/requirements.txt

# Run the service
uvicorn production_dev.main:app --host 0.0.0.0 --port 9742 --reload

# Open in browser: http://localhost:9742
```

---

## 📁 Project Structure

```
ostrofun/
├── 📂 production_dev/          # 🚀 PRODUCTION SERVICE (this is what you run)
│   ├── main.py                 # FastAPI application
│   ├── predictor.py            # Core prediction logic
│   ├── cache_service.py        # Memory cache management
│   ├── generate_cache.py       # 🆕 Dual-model cache generator
│   ├── train_full_model.py     # 🆕 Full model training script
│   ├── daily_retrain.py        # 🆕 Daily retraining automation
│   ├── data_service.py         # Database data fetching
│   ├── schemas.py              # API request/response models
│   ├── static/                 # Web UI files
│   │   ├── index.html          # Main page
│   │   ├── styles.css          # Styling
│   │   └── app.js              # Frontend logic
│   ├── Dockerfile              # Docker build instructions
│   ├── docker-compose.yml      # Docker deployment config
│   └── requirements.txt        # Python dependencies
│
├── 📂 RESEARCH/                # 🔬 RESEARCH & TRAINING (for model development)
│   ├── astro_engine.py         # Swiss Ephemeris calculations
│   ├── features.py             # Feature engineering
│   ├── model_training.py       # XGBoost training utilities
│   ├── data_loader.py          # PostgreSQL data loading
│   ├── labeling.py             # Price movement labeling
│   ├── birthdate_deep_search.ipynb  # Model hyperparameter tuning
│   ├── xgb_hyperparam_search.py     # Grid search for best params
│   └── research_timeline.md    # 📋 Complete research history
│
├── 📂 RESEARCH-REPRODUCE/      # 🔄 REPRODUCIBILITY PACKAGE
│   ├── README.md               # Step-by-step reproduction guide
│   ├── reproduce_all.py        # Master script to reproduce 60.3%
│   ├── step1_baseline/         # Initial pipeline files
│   ├── step2_grid_search/      # Grid search files
│   ├── step3_body_ablation/    # Body ablation study
│   ├── step4_birthdate_search/ # Birth date search
│   └── step5_deep_tuning/      # Final 60.3% configuration
│
├── 📂 src/                     # 📦 CORE LIBRARY
│   └── models/
│       └── xgb.py              # XGBBaseline model class
│
├── 📂 configs/                 # ⚙️ CONFIGURATION FILES
│   ├── astro.yaml              # Astrological settings
│   ├── db.yaml                 # Database connection
│   └── subjects.yaml           # Trading pairs config
│
├── 📂 models_artifacts/        # 💾 SAVED MODELS
│   ├── btc_astro_predictor.joblib      # 🆕 Split model (backtest)
│   └── btc_astro_predictor.full.joblib # 🆕 Full model (forecast)
│
├── 📂 data/                    # 📊 DATA FILES
│   ├── ephe/                   # Swiss Ephemeris data files
│   ├── prediction_cache/       # 🆕 Cached predictions (parquet)
│   └── processed/              # Processed datasets
│
└── 📄 README.md                # You are here!
```

---

## 💻 Installation

### Prerequisites

1. **Python 3.11+** - [Download Python](https://www.python.org/downloads/)
2. **PostgreSQL** - For storing market data
3. **Docker Desktop** (optional) - [Download Docker](https://www.docker.com/products/docker-desktop/)

### Step-by-Step Installation

```bash
# 1. Clone or navigate to the project
cd /home/rut/ostrofun

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/WSL
# or: .\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r production_dev/requirements.txt

# 4. Configure database connection
# Edit configs/db.yaml with your PostgreSQL credentials

# 5. Train the model (or use pre-trained)
cd RESEARCH
python -c "from model_training import train_xgb_model; print('Ready!')"
```

---

## 🌐 Running the Prediction Service

### Local Development

```bash
# Start the FastAPI server
cd /home/rut/ostrofun
uvicorn production_dev.main:app --host 0.0.0.0 --port 9742 --reload

# The service will be available at:
# - Web UI: http://localhost:9742
# - API Docs: http://localhost:9742/api/docs
# - Health Check: http://localhost:9742/api/health
```

### Docker Production

```powershell
# From Windows PowerShell
cd \\wsl$\Ubuntu-24.04\home\rut\ostrofun

# Build the image
docker build -t btc-astro-predictor -f production_dev/Dockerfile .

# Run the container
docker run -d -p 9742:9742 --name btc-predictor btc-astro-predictor

# Check status
docker logs btc-predictor

# Stop when done
docker stop btc-predictor
```

---

## 📡 API Reference

### GET `/api/health`

Check if the service is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-02-03T12:00:00",
  "model_loaded": true,
  "natal_date": "2009-10-10",
  "expected_accuracy": 0.603
}
```

### GET `/api/predict?days=90`

Generate price direction predictions.

**Parameters:**
- `days` (optional): Number of days to predict (1-365, default: 90)
- `seed` (optional): Random seed for reproducible price simulation

**Response:**
```json
{
  "predictions": [
    {
      "date": "2024-02-04",
      "direction": "UP",
      "confidence": 0.65,
      "simulated_price": 105234.56
    }
  ],
  "summary": {
    "total_days": 90,
    "up_predictions": 52,
    "down_predictions": 38,
    "up_ratio": 0.578,
    "average_confidence": 0.612
  }
}
```

### GET `/api/historical?days=30`

Get historical BTC prices from database.

**Parameters:**
- `days` (optional): Number of historical days (1-365, default: 30)

---

## 🎓 Training Your Own Model

### 1. Prepare Market Data

Ensure your PostgreSQL database has the `market_daily` table with BTC price data:

```sql
CREATE TABLE market_daily (
    date DATE NOT NULL,
    subject_id VARCHAR(50) NOT NULL,
    close NUMERIC(18, 8) NOT NULL,
    PRIMARY KEY (date, subject_id)
);
```

### 2. Run the Training Notebook

```bash
cd /home/rut/ostrofun/RESEARCH

# Open Jupyter
jupyter notebook birthdate_deep_search.ipynb
```

### 3. The notebook will:

1. Load market data from your database
2. Create binary labels (UP/DOWN based on price change)
3. Calculate astrological features using Swiss Ephemeris
4. Train multiple XGBoost models with different parameters
5. Find the best model configuration
6. Save the trained model to `models_artifacts/`

### 4. Current Best Configuration

| Parameter | Value |
|-----------|-------|
| Birth Date | 2009-10-10 |
| Coordinate Mode | both (geo + helio) |
| Orb Multiplier | 0.1 |
| Gauss Window | 200 |
| Gauss Std | 70.0 |
| XGBoost Trees | 500 |
| Max Depth | 6 |
| Learning Rate | 0.03 |
| Colsample | 0.6 |
| **R_MIN Score** | **0.603** |
| **MCC Score** | **0.315** |

---

## 📜 Research Timeline

The model was developed through a systematic research process over 2 days (2026-02-02 → 2026-02-03):

### Metric Progression

| Phase | R_MIN | MCC | Key Discovery |
|-------|-------|-----|---------------|
| Baseline | 50.0% | 0.0 | Random guessing |
| Initial XGB | 50.4% | 0.06 | Grid search started |
| Body Ablation | 55.7% | 0.12 | Uranus+Pluto = noise |
| Param Tuning | 57.9% | 0.16 | Tight orbs (0.05) work |
| +Natal Transits | 59.0% | 0.19 | orb=0.075, win=200 |
| **Deep Tuning** | **60.3%** | **0.315** | birth=2009-10-10 |

### Key Research Files

| File | Purpose |
|------|---------|
| `RESEARCH/research_timeline.md` | Complete research history |
| `RESEARCH/birthdate_deep_search.ipynb` | Final 60.3% model |
| `RESEARCH/xgb_hyperparam_search.ipynb` | XGBoost tuning |
| `RESEARCH/body_ablation_research.ipynb` | Which bodies matter |

📋 **Full documentation:** See [`RESEARCH/research_timeline.md`](RESEARCH/research_timeline.md)

---

## 🔄 Reproducibility

All research results are fully reproducible! We provide a complete package to recreate the 60.3% recall model from scratch.

### Quick Reproduction

```bash
cd RESEARCH-REPRODUCE
conda activate btc
python reproduce_all.py
```

### Step-by-Step Reproduction

The `RESEARCH-REPRODUCE/` folder contains 5 steps:

| Step | Script | Expected R_MIN |
|------|--------|----------------|
| 1 | `step1_baseline/main_pipeline.py` | ~50% |
| 2 | `step2_grid_search/grid_search.py` | ~52-55% |
| 3 | `step3_body_ablation/xgb_hyperparam_search.py` | ~57-58% |
| 4 | `step4_birthdate_search/birthdate_search.py` | ~58% |
| 5 | `step5_deep_tuning/birthdate_deep_search.py` | **60.3%** |

### Verification Checklist

- [ ] Step 1: R_MIN ≈ 0.50 (baseline)
- [ ] Step 3: R_MIN ≈ 0.58 (body ablation)
- [ ] Step 5: R_MIN = 0.603, MCC = 0.315 (FINAL)

📦 **Full guide:** See [`RESEARCH-REPRODUCE/README.md`](RESEARCH-REPRODUCE/README.md)

---

## ⚙️ Configuration

### Database (`configs/db.yaml`)

```yaml
database:
  url: "postgresql://user:password@localhost:5432/ostrofun"
```

### Astrology Settings (`configs/astro.yaml`)

```yaml
bodies:
  - Sun
  - Moon
  - Mercury
  - Venus
  - Mars
  - Jupiter
  - Saturn
  - Uranus
  - Neptune
  - Pluto

aspects:
  - name: conjunction
    angle: 0
    orb: 10
  - name: sextile
    angle: 60
    orb: 6
  # ... etc
```

---

## ❓ Frequently Asked Questions

### Q: Does astrology really affect Bitcoin prices?

**A:** This is an experimental project! Our model achieves ~60% accuracy, which is better than random chance (50%), suggesting there may be some correlation. However, correlation does not equal causation. Use this for educational and entertainment purposes only.

### Q: What is R_MIN?

**A:** R_MIN (Recall Minimum) is the minimum recall between the UP and DOWN classes. A model with 0.60 R_MIN correctly identifies at least 60% of both up days AND down days. This ensures the model isn't biased toward one direction.

### Q: Why October 10, 2009?

**A:** This is the date when the first Bitcoin exchange rate was established by New Liberty Standard (1,309.03 BTC = $1.00). We consider this Bitcoin's "economic birth" rather than the genesis block date (January 3, 2009).

### Q: Can I use this for trading?

**A:** **NO!** This is an experimental research project. Never trade based solely on this model. Past performance does not guarantee future results. You could lose money.

### Q: How do I update the market data?

**A:** The service fetches data from your PostgreSQL database. Keep your `market_daily` table updated with the latest BTC prices.

---

## ⚠️ Disclaimer

> **THIS IS NOT FINANCIAL ADVICE.**
>
> This project is for **educational and entertainment purposes only**. Cryptocurrency trading involves substantial risk of loss. The predictions made by this software are based on experimental astrological correlations and should not be used as the sole basis for investment decisions.
>
> Past performance does not guarantee future results. Always do your own research and consider consulting a qualified financial advisor before making investment decisions.
>
> The creators of this project are not responsible for any financial losses incurred from using this software.

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

---

## 🙏 Credits

- **Swiss Ephemeris** - High-precision astronomical calculations
- **XGBoost** - Gradient boosting framework
- **FastAPI** - Modern Python web framework
- **Chart.js** - Interactive charts

---

*Made with ☿ Mercury retrograde energy 🌙*
