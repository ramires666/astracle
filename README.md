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

## Separate quotes download (1d)

This command uses `configs/market.yaml` and prints verbose logs/progress:

```
python scripts/download_klines_1d.py
```

Example with explicit params:

```
python scripts/download_klines_1d.py --symbol BTCUSDT --market-type futures_um --data-root data/market --auto-start --progress
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
