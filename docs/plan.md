# Project plan: astro price direction prediction (MVP -> expansion)

This document фиксес architecture, pipeline stages, and conventions.
All new files are created **outside** `old-project`. Files in `old-project`
are read and reused but not modified.

---

## 1) Goal and MVP scope

**Goal:** validate the hypothesis that astro data alone can signal price direction.

**MVP constraints:**
- timeframe: **1 day (1d)**
- market data is used **only** for labels (oracle)
- houses/cusps **not used** (yet)
- features: body positions + aspects (major)
- model: XGBoost baseline

---

## 2) Terms

- **subject_id** - object identifier (BTC, SPX, GOLD, etc.)
- **birth** - object "birth" timestamp (configured)
- **astro_bodies_daily** - body positions per day
- **astro_aspects_daily** - aspects between bodies per day
- **natal** - natal chart of the object
- **transits** - current bodies' aspects to natal chart

---

## 3) Data sources

### 3.1 Market data
- source: **Binance Vision (public archive)**
- use **1d** klines (fast and enough for MVP)
- auto-search the **earliest** available date in the archive
- store only **close**

### 3.2 Astro data
- Swiss Ephemeris (sweph)
- standard planets + Lunar nodes + Black Moon
- no Chiron
- aspects: **major** (0/60/90/120/180)

---

## 4) Project structure (new files)

```
configs/
  subjects.yaml
  market.yaml
  astro.yaml
  labels.yaml
docs/
  plan.md
scripts/
src/
  common/
    config.py
  market/
    downloader.py
    parser.py
    loader.py
  astro/
    config/
      bodies.yaml
      aspects.yaml
    engine/
      models.py
      settings.py
      calculator.py
      aspects.py
  db/
    schema.sql
    connection.py
  pipeline/
    mvp.py
  features/
    builder.py
  labeling/
    oracle.py
  models/
    xgb.py
  visualization/
    dxcharts_report.py
```

---

## 5) DB schema (Postgres/Timescale)

All tables are tied to **subject_id**.

1) `subjects`
- `subject_id` (PK)
- `symbol`, `exchange`
- `birth_dt_utc`, `birth_lat`, `birth_lon`

2) `market_daily`
- `subject_id`
- `date`
- `close`

3) `astro_bodies_daily`
- `subject_id`, `date`
- `body`, `lon`, `lat`, `speed`, `is_retro`, `sign`, `declination`

4) `astro_aspects_daily`
- `subject_id`, `date`
- `p1`, `p2`, `aspect`, `orb`, `is_exact`, `is_applying`

5) `natal_bodies`
- `subject_id`, `body`, `lon`, `lat`, `speed`, `is_retro`, `sign`, `declination`

6) `natal_aspects`
- `subject_id`, `p1`, `p2`, `aspect`, `orb`

7) `transit_aspects_daily`
- `subject_id`, `date`
- `transit_body`, `natal_body`, `aspect`, `orb`, `is_exact`, `is_applying`

8) `features_daily`
- `subject_id`, `date`, features in wide format

9) `labels_daily`
- `subject_id`, `date`
- `target`, `smoothed_close`, `smooth_slope`

---

## 6) Pipeline (MVP)

1) **Download market 1d**
   - auto-start: first available date in public archive

2) **Parse -> market_daily**
   - keep only `date`, `close`

3) **astro_bodies_daily**
   - body positions per day

4) **astro_aspects_daily**
   - aspects between bodies (major)

5) **features_daily**
   - wide astro feature table

6) **labels_daily (oracle)**
   - Gaussian smoothing of price, slope -> labels

7) **XGB baseline**
   - astro features only
   - time-based train/val/test split

---

## 7) Visual checks

To validate correctness:
- price vs smoothed_close
- labels (down/sideways/up)
- class distribution

**Charts via dxcharts** (or similar).
Docs via **MCP Context7**.

---

## 8) Expansion (after MVP)

1) `natal_bodies` + `natal_aspects`
2) `transit_aspects_daily`
3) Progressions/directions
4) XGB tuning and feature importance analysis
