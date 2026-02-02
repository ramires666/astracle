# ostrofun

## Quick start (Windows/Conda)

1) Create or update the environment from the file:

```
conda env create -f environment.yml -n crypto_analysis_env
```

If the environment already exists:

```
conda env update -f environment.yml -n crypto_analysis_env --prune
```

2) Activate:

```
conda activate crypto_analysis_env
```

3) If needed, sync pip packages (pip layer):

```
python -m pip install -r requirements.txt
```

4) Check key dependencies:

```
python -c "import swisseph, svgwrite; print('ok')"
```

5) Smoke test:

```
python scripts/smoke_check.py
```

6) Run example:

```
python main.py
```

## Astro balanced pipeline (step-by-step, cached)

This replaces the large notebook `notebooks/astro_xgboost_training_balanced.ipynb`.
Each step writes a parquet (and a `.meta.json` cache file). If inputs/params are
unchanged, the step is skipped automatically unless you pass `--force`.

Pipeline CLI:

```
python scripts/astro_pipeline.py
```

Step-by-step (recommended for debugging):

1) Market data (DB-first, then web if needed)

```
python scripts/astro_pipeline.py --step market --market-update
```

2) Labels (balanced UP/DOWN)

```
python scripts/astro_pipeline.py --step labels
```

3) Astro bodies/aspects

```
python scripts/astro_pipeline.py --step astro
```

4) Astro features

```
python scripts/astro_pipeline.py --step features
```

5) Merge dataset (features + labels)

```
python scripts/astro_pipeline.py --step dataset --write-inventory
```

6) Time split

```
python scripts/astro_pipeline.py --step split
```

7) Train model

```
python scripts/astro_pipeline.py --step train
```

Training uses hyperparameter tuning by default (RandomizedSearchCV + TimeSeriesSplit).
You can disable it in `configs/training.yaml` (`tune_enabled: false`) if you want faster runs.

8) Eval plots (confusion matrix + feature importance)

```
python scripts/astro_pipeline.py --step eval
```

### Orb grid search (aspect orb size)

Run a grid over orb multipliers (uses current labels from config):

```
python scripts/astro_orb_grid.py --orb-list 0.8,1.0,1.2 --force
```

Results saved to:
- `data/market/reports/orb_grid_summary.csv`
- `data/market/reports/orb_grid_summary.parquet`

### Geo + Helio features together

If you want to include both coordinate centers and merge their features:

```
python scripts/astro_pipeline.py --include-both-centers --from-step astro --to-step train --force
```

This will generate feature columns with prefixes `geo__` and `helio__`.

Run a range:

```
python scripts/astro_pipeline.py --from-step market --to-step eval --market-update
```

Important flags:
- `--force` re-runs a step even if cache is valid.
- `--market-update` forces checking/downloading market data.
- `--save-db/--no-save-db` controls writing market_daily to DB.
- `--label-mode` (balanced_detrended or balanced_future_return)
- `--label-price-mode` (raw or log)
- `--horizon`, `--gauss-window`, `--gauss-std`, `--target-move-share`
- `--label-report-sample-days` (downsample HTML report payload; 1 = full)
- `--label-report-max-points` (cap HTML report points; 0 = no cap)
- `--orb-multiplier` (aspects orb scaling)
- `--include-transit-aspects` (transit->natal)

Outputs (default):
- Market: `data/market/processed/<subject>_market_daily.parquet`
- Labels: `data/market/processed/<subject>_labels_*.parquet`
- Bodies/Aspects: `data/market/processed/<subject>_astro_bodies.parquet`, `..._astro_aspects_*.parquet`
- Features: `data/market/processed/<subject>_features_*.parquet`
- Dataset: `data/market/processed/<subject>_dataset_*.parquet`
- Splits: `data/market/processed/<subject>_split_*_{train,val,test}.parquet`
- Model: `models_artifacts/xgb_astro_*.joblib`
- Reports: `data/market/reports/*.json|png`

## Control Center (FastAPI + React)

The project includes a lightweight local dashboard to run pipeline steps, view logs,
and open reports without memorizing CLI flags.

1) Start the API server:

```
python -m src.app.api
```

The API listens on `http://127.0.0.1:8001`.

2) Start the frontend (Vite):

```
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in your browser.

## Market download behavior (DB-first)

The market step checks local DB first. If DB is up-to-date, no download happens.
If DB is missing or behind, it downloads monthly archives and daily "tail" files.

Config keys in `configs/market.yaml`:
- `use_db_cache: true` (read DB first)
- `write_db: true` (write market_daily into DB after build)

## Separate quotes download (1d)

This command uses `configs/market.yaml` and prints verbose logs/progress:

```
python scripts/download_klines_1d.py
```

Example with explicit params:

```
python scripts/download_klines_1d.py --symbol BTCUSDT --market-type futures_um --data-root data/market --auto-start --progress
```

DB-first (skip download if DB cache exists):

```
python scripts/download_klines_1d.py --check-db
```

Force download even if DB has data:

```
python scripts/download_klines_1d.py --force
```

## Postgres in Docker (data inside project)

Start DB:

```
docker compose up -d
```

Stop DB:

```
docker compose down
```

DB data will be stored in:

```
data/postgres
```

Default connection string:

```
postgresql://ostro:ostro_pass@localhost:55432/ostrofun
```

## Load quotes and astro data into DB

1) Make sure Postgres is running:

```
docker compose up -d
```

2) Download quotes and parse to parquet:

```
python scripts/download_klines_1d.py --symbol BTCUSDT --market-type futures_um --data-root data/market --auto-start --progress
```

3) Load quotes and astro data into DB:

```
python scripts/load_to_db.py --progress
```

## Archived BTC quotes (full OHLCV)

Parse archived file `Bitcoin Historical Data.csv`:

```
python scripts/parse_bitcoin_archive.py
```

Results:
- `data/market/processed/BTC_archive_ohlcv_daily.parquet` - full OHLCV
- `data/market/processed/BTC_archive_market_daily.parquet` - only date/close (for DB)

If you want to load only the archived close file into DB:

```
python scripts/load_to_db.py --market-parquet data/market/processed/BTC_archive_market_daily.parquet --progress
```

## Merge archived and fresh data (max full OHLCV)

The archived CSV ends around 2024-03-24; fresh data comes from Binance.
Command below merges archive + Binance into one dataset:

```
python scripts/merge_btc_ohlcv.py --progress
```

Results:
- `data/market/processed/BTC_full_ohlcv_daily.parquet` - full OHLCV
- `data/market/processed/BTC_full_market_daily.parquet` - only date/close (for DB)

Load merged set into DB:

```
python scripts/load_to_db.py --market-parquet data/market/processed/BTC_full_market_daily.parquet --progress
```

## Where to set subject birth datetime

File `configs/subjects.yaml`, field `birth_dt_utc` (UTC):

```
subjects:
  - subject_id: btc
    birth_dt_utc: "2010-07-17T00:00:00Z"
```

## Recompute astro data after birth date change

After editing `birth_dt_utc`:

1) Update DB and recompute natal data:

```
python scripts/load_to_db.py --natal --progress
```

2) If you need transit-to-natal aspects:

```
python scripts/load_to_db.py --natal --transits --progress
```

Note: daily transits (astro_bodies_daily/astro_aspects_daily) do not depend on birth date,
but natal and transit-to-natal do.

## Environment files

- `environment.yml` - main conda snapshot (with pip section).
- `requirements.txt` - minimal pip layer (not in conda-forge on Windows).
- `requirements.in` - minimal pip list not in conda-forge on Windows.

## Notes

- On Windows, `pyswisseph` is not available in conda-forge, so install via pip (already in `requirements.txt`).
- Import module name is `swisseph`, not `pyswisseph`.
