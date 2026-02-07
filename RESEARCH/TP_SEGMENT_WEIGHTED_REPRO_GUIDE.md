# TP Segment-Weighted Final Notebook Repro Guide

## Goal
Reproduce the final result from:
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`

including the final aggregate conclusion:
- `Verdict (weighted-hit): aggregate evidence supports non-random TP-segment alignment.`

This guide is based on the actual git history and current notebook configuration.

## 1) What git history shows (required lineage)

The relevant evolution chain is:

1. `f4940e4` (2026-02-06)
- Added base turning-point grid notebook:
  - `RESEARCH/grid_search_massive_label_weighted.ipynb`
- Added turning-point modules in `RESEARCH2/Moon_cycles/*`.

2. `2f9aaab` (2026-02-07)
- Updated `RESEARCH/grid_search_massive_label_weighted.ipynb`
- Commit message indicates massive grid run completion.

3. `bd8bfdc` (2026-02-07)
- Added post-analysis notebook:
  - `RESEARCH/grid_search_massive_label_weighted_analysis.ipynb`
- Added first significance notebook:
  - `RESEARCH/grid_search_massive_label_weighted_statistical_significance.ipynb`

4. `ec2fd41` (2026-02-07)
- Added stricter significance variants:
  - `RESEARCH/grid_search_massive_label_weighted_statistical_significance_soft_regime.ipynb`
  - `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_points_only.ipynb`

5. Final notebook used now:
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`
- This is the TP-segment-weighted extension over the above lineage.

## 2) Minimal notebooks to execute

There are two practical paths.

### Path A: Fast reproduction (recommended)
Run only:
1. `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`

Requirements for Path A:
- Existing checkpoint and top-N stats files must already exist (see section 4).
- Then no massive grid rerun is needed.

### Path B: Full from-scratch reproduction
Run in this order:
1. `RESEARCH/grid_search_massive_label_weighted.ipynb`
- Produces core checkpoint of the massive search.
2. `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`
- Re-evaluates top candidates and computes TP segment-weighted significance.

Optional (not required for final verdict, but useful for sanity checks):
- `RESEARCH/grid_search_massive_label_weighted_analysis.ipynb`
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_soft_regime.ipynb`
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_points_only.ipynb`

## 3) Environment and dependencies (minimum)

From project root:

```bash
conda activate btc
pip install -r RESEARCH/requirements.txt
```

Important:
- `seaborn` must be installed (it is imported from `RESEARCH2/Moon_cycles/eval_visuals.py`).
- `pyswisseph`, `xgboost`, `scikit-learn`, `pyarrow`, `matplotlib` are required.

## 4) Data/artifacts that must exist

### Market parquet (required)
At least one of these must exist:
- `data/market/processed/BTC_full_market_daily.parquet` (preferred)
- `data/market/processed/BTCUSDT_market_daily.parquet`
- `data/market/processed/btc_market_daily.parquet`
- `data/market/processed/BTC_archive_market_daily.parquet`

### Fast-path report artifacts (required for Path A)
- `data/market/reports/turning_massive_label_grid_checkpoint.csv`
- `data/market/reports/turning_massive_label_grid_top100_event_stats_true-global_turning_points_pred-hard_label_tp9_gap14_seg10.csv`
- `data/market/reports/turning_massive_label_grid_top100_tp_segment_weighted_g1p5_mind5_tail1_pred-hard_label.csv`

Quick check:

```bash
ls -lh \
  data/market/reports/turning_massive_label_grid_checkpoint.csv \
  data/market/reports/turning_massive_label_grid_top100_event_stats_true-global_turning_points_pred-hard_label_tp9_gap14_seg10.csv \
  data/market/reports/turning_massive_label_grid_top100_tp_segment_weighted_g1p5_mind5_tail1_pred-hard_label.csv
```

If market parquet is missing, build it once:

```bash
python scripts/merge_btc_ohlcv.py --progress
```

## 5) Critical config in final notebook

In `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb` defaults are:
- `RUN_TAG = "turning_massive_label_grid"`
- `TRUE_EVENT_MODE = "global_turning_points"`
- `PRED_EVENT_MODE = "hard_label"`
- `TOP_N = 100`
- `SEGMENT_TOP_N = 100`
- `SEGMENT_SCORE_GAMMA = 1.5`
- `SEGMENT_MIN_DAYS = 5`
- `SEGMENT_INCLUDE_OPEN_TAIL = True`

### To avoid expensive recomputation in Path A
Set before running heavy cells:
- `RUN_TOP100_NOW = False`
- `RUN_TOP100_SEGMENT_NOW = False`

Then notebook will load existing CSV artifacts instead of retraining/evaluating top-N again.

## 6) Execution recipe (Path A, minimal)

1. Open
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`

2. Set fast flags in config cells:
- `RUN_TOP100_NOW = False`
- `RUN_TOP100_SEGMENT_NOW = False`

3. Run all cells top-to-bottom.

4. Verify outputs:
- Global summary cell prints top-N significance diagnostics.
- Verdict line appears:
  - `Verdict (weighted-hit): aggregate evidence supports non-random TP-segment alignment.`
- Extended interpretation and publication visuals at notebook end render successfully.

## 7) Execution recipe (Path B, full)

1. Open and run:
- `RESEARCH/grid_search_massive_label_weighted.ipynb`

Expected output artifacts:
- `data/market/reports/turning_massive_label_grid_checkpoint.csv`
- `data/market/reports/turning_massive_label_grid_done_pairs.txt`

2. Open and run:
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`

Use defaults if you want full recomputation:
- `RUN_TOP100_NOW = True`
- `RUN_TOP100_SEGMENT_NOW = True`

Expected additional artifacts:
- `data/market/reports/turning_massive_label_grid_top100_event_stats_true-global_turning_points_pred-hard_label_tp9_gap14_seg10.csv`
- `data/market/reports/turning_massive_label_grid_top100_tp_segment_weighted_g1p5_mind5_tail1_pred-hard_label.csv`

## 8) Notes to avoid common failures

1. If you see `ModuleNotFoundError: seaborn`:
- Install `RESEARCH/requirements.txt` in the active kernel env.

2. If charts block execution:
- Keep `PLOT_BLOCKING = False` in notebook config.

3. If you have checkpoint files and do not want rerun:
- Keep both `RUN_TOP100_NOW` and `RUN_TOP100_SEGMENT_NOW` set to `False`.

4. If checkpoint schema changed between notebook versions:
- Remove old incompatible checkpoint files and regenerate with the current notebook version.

## 9) Minimal reproducibility checklist

- [ ] `btc` kernel/environment selected
- [ ] `pip install -r RESEARCH/requirements.txt` done
- [ ] market parquet exists in `data/market/processed/`
- [ ] `turning_massive_label_grid_checkpoint.csv` exists
- [ ] top100 event and segment CSV artifacts exist (for fast path)
- [ ] final notebook run completes without missing imports
- [ ] verdict line is printed in global summary cell
