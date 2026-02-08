# Research History: Astro Tabular NN (Best-Grid Labeling)

## 2026-02-08 - Switch to best-grid labeling from latest TP notebook

### Goal
Use the exact labeled dataset logic from:
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`

and run a dual-network neural trial (DCN + DeepFM) on that same markup.

### What was changed in code
1. Added exact dataset reconstruction utility:
- `RESEARCH/astro_tabular_nn/best_grid_dataset.py`
- Source checkpoint: `data/market/reports/turning_massive_label_grid_checkpoint.csv`
- Ranking logic copied from notebook (`sort_results_frame` ordering)
- Top row selected as `grid_best`
- Feature build path matches notebook:
  - `build_turning_astro_feature_set(..., cache_namespace="research2_turning_grid")`
  - `label_turning_points(...)`
  - `build_turning_target_frame(...)`
  - `merge_features_with_turning_target(...)`

2. Added second neural architecture:
- `RESEARCH/astro_tabular_nn/model_deepfm.py` (DeepFM-style interaction model)

3. Extended trainer/metrics for binary + ternary tasks:
- `RESEARCH/astro_tabular_nn/metrics_numba.py`
- `RESEARCH/astro_tabular_nn/trainer.py`
- Binary path now uses threshold search, ternary path keeps margin search.

4. Extended scout and grid to two-network mode:
- `RESEARCH/astro_tabular_nn/experiments.py`
- `RESEARCH/astro_tabular_nn/grid_search.py`
- `RESEARCH/astro_tabular_nn/grid_trial.py`
- `RESEARCH/astro_tabular_nn/quick_scout.py`

5. Export/docs updates:
- `RESEARCH/astro_tabular_nn/__init__.py`
- `RESEARCH/astro_tabular_nn/README.md`
- `RESEARCH/astro_tabular_nn/astro_tabular_nn_quick_scout.ipynb`
- `RESEARCH/astro_tabular_nn/astro_tabular_nn_grid_trial.ipynb`

6. Leakage fix after first probe run:
- Updated `DatasetConfig.drop_cols` in `RESEARCH/astro_tabular_nn/config.py`
- Explicitly excluded non-feature columns from model input:
  - `next_ret`, `turning_direction`, `sample_weight`, `target_mode`, `event_index`, `segment_index`
- This aligned NN feature selection with the source notebook logic and removed target leakage.

### Exact best-grid dataset built
Command logic used via module call:
- `build_best_grid_labeled_dataset(run_tag="turning_massive_label_grid", data_start="2017-11-01")`

Resulting cached parquet:
- `RESEARCH/cache/astro_tabular_nn_best_grid__dataset__0faeef66.parquet`

Resolved top checkpoint row (grid_best):
- `eval_id=7795`
- `target_mode=segment_midpoint`
- `threshold=0.61`
- `feature_coord_mode=both`
- `feature_orb_mult=0.1`
- `birth_dt_utc=2009-10-10T18:15:05Z`

Dataset stats:
- shape: `3009 x 2678`
- target counts: `{0: 1865, 1: 1144}`
- date range: `2017-11-12 .. 2026-02-06`

### Test scout run (dual-network smoke)
Initial smoke run was executed before the leakage fix and is kept only for traceability.

Pre-fix command:
```bash
python -m RESEARCH.astro_tabular_nn.quick_scout \
  --dataset-source best-grid \
  --run-tag turning_massive_label_grid \
  --data-start 2017-11-01 \
  --epochs 3 \
  --batch-size 512 \
  --seeds 42 \
  --out-csv RESEARCH/reports/astro_tabular_nn_quick_scout_best_grid_e3_s1.csv
```

Pre-fix output file:
- `RESEARCH/reports/astro_tabular_nn_quick_scout_best_grid_e3_s1.csv`

Post-fix (valid) command:
```bash
python -m RESEARCH.astro_tabular_nn.quick_scout \
  --dataset-source best-grid \
  --run-tag turning_massive_label_grid \
  --data-start 2017-11-01 \
  --epochs 3 \
  --batch-size 512 \
  --seeds 42 \
  --out-csv RESEARCH/reports/astro_tabular_nn_quick_scout_best_grid_e3_s1_noleak.csv
```

Output file:
- `RESEARCH/reports/astro_tabular_nn_quick_scout_best_grid_e3_s1_noleak.csv`

Observation:
- After leakage fix, metrics dropped to realistic values.
- In this short scout, both architectures struggled on rare class recall in the latest test tail.

### Probe broad grid (two networks) on best-grid markup
Pre-fix command:
```bash
python -m RESEARCH.astro_tabular_nn.grid_trial \
  --dataset-source best-grid \
  --run-tag turning_massive_label_grid \
  --data-start 2017-11-01 \
  --epochs 4 \
  --batch-size 512 \
  --n-trials 24 \
  --model-types dcn deepfm \
  --seeds 42 \
  --sample-seed 20260208 \
  --out-csv RESEARCH/reports/astro_tabular_nn_grid_trial_best_grid_e4_t24_s1.csv
```

Initial broad trial was executed before the leakage fix and is kept only for traceability:
- `RESEARCH/reports/astro_tabular_nn_grid_trial_best_grid_e4_t24_s1.csv`

Post-fix (valid) command:
```bash
python -m RESEARCH.astro_tabular_nn.grid_trial \
  --dataset-source best-grid \
  --run-tag turning_massive_label_grid \
  --data-start 2017-11-01 \
  --epochs 4 \
  --batch-size 512 \
  --n-trials 24 \
  --model-types dcn deepfm \
  --seeds 42 \
  --sample-seed 20260208 \
  --out-csv RESEARCH/reports/astro_tabular_nn_grid_trial_best_grid_e4_t24_s1_noleak.csv
```

Post-fix output file:
- `RESEARCH/reports/astro_tabular_nn_grid_trial_best_grid_e4_t24_s1_noleak.csv`

Top run from post-fix probe grid:
- `model_type=dcn`
- `hidden_dims=512x256x128`
- `cross_layers=2`
- `cross_rank=32`
- `dropout=0.25`
- `learning_rate=0.0005`
- `weight_decay=5e-4`
- `class_weight_power=1.4`
- `label_smoothing=0.05`
- `batch_size=768`
- `cutoff_kind=threshold`
- `best_margin=0.60` (threshold for binary task)
- `test_recall_min=0.4444`
- `test_mcc=0.0063`

### Repro checklist (to get same direction again)
1. Ensure CUDA is available and active.
2. Keep checkpoint file unchanged:
   - `data/market/reports/turning_massive_label_grid_checkpoint.csv`
3. Build/load dataset via `--dataset-source best-grid` (or `ensure_best_grid_dataset_path`).
4. Use fixed seeds in commands.
5. Use the same `sample-seed` for candidate sampling in grid trial.

### Important caveat
Current chronological split has highly imbalanced test tail for this markup (`test target share ≈ 98% class 0`).
This heavily limits reliability of one-split leaderboard metrics.
For robust model ranking, repeat with additional time splits / walk-forward folds in the next iteration.

## 2026-02-08 - Wide scout expansion with balance-focused objective

### Goal
Run a significantly wider scout around DCN/DeepFM settings with explicit emphasis on:
- maximize `test_recall_min`
- minimize `test_recall_gap`

### Run configuration
Dataset source remained the same:
- `/home/rut/ostrofun/RESEARCH/cache/astro_tabular_nn_best_grid__dataset__0faeef66.parquet`

Balance-oriented sweep settings:
- threshold grid: `0.10 .. 0.90` (step `0.01`)
- `margin_gap_penalty=0.70`
- `margin_prior_penalty=0.12`
- epochs: `6`
- seeds: `4`
- model specs: `20`
- total runs: `80`

Main artifact:
- `RESEARCH/reports/astro_tabular_nn_quick_scout_wide_balance_e6_s4_20260208_055644.csv`
- `RESEARCH/reports/astro_tabular_nn_quick_scout_wide_balance_e6_s4_20260208_055644.meta.json`

### Notebook-style post-run ranking package
Generated ranking/diagnostics package:
- `RESEARCH/reports/astro_tabular_nn_postrun_wide_balance_20260208_055915/`
- key files:
  - `REPORT.md`
  - `selected_models.csv`
  - `leaderboard_top15.csv`
  - `ranked_results.csv`
  - per-model diagnostics PNG + split metrics CSV for top-5

Selected winner from this stage:
- model: `dcn_dropout_high` (`dcn`)
- seed: `45`
- test: `recall_min=0.6682`, `recall_gap=0.1096`, `MCC=0.1314`, `ACC=0.6704`

This model was used as the baseline winner for follow-up focused search.

## 2026-02-08 - Price vs truth event-alignment charts (old-notebook style)

### Goal
Reproduce old notebook style visual QA:
- test-period BTC price chart
- true turning events
- predicted regime/switch overlay
- event matching metrics in a fixed window

Reference notebook for visual style intent:
- `RESEARCH/grid_search_massive_label_weighted_statistical_significance_tp_segment_weighted.ipynb`

### Code additions and refactor
1. Added event alignment helper module:
- `RESEARCH/astro_tabular_nn/event_alignment.py`
- includes:
  - regime switch extraction
  - global TP-based true-event build via `label_turning_points`
  - greedy one-to-one event matching in `±window` days
  - event precision/recall/lag metrics
  - old-notebook-like plotting helper

2. Added report CLI for price-event comparison:
- `RESEARCH/astro_tabular_nn/price_event_report.py`
- supports:
  - latest `selected_models.csv` auto-pick
  - top-k model replay
  - true event modes (`global_turning_points` / `target_switch`)
  - prediction regime modes (`hard_label` / probability-based smoothed regimes)
  - export of all intermediate CSVs plus PNG charts and markdown report

3. Extended trainer output for downstream diagnostics:
- `RESEARCH/astro_tabular_nn/trainer.py`
- `FitResult` now optionally carries:
  - `test_pred`
  - `test_proba`
- enabled by `capture_predictions=True` in replay/report pipelines.

### Generated report package
- `RESEARCH/reports/astro_tabular_nn_price_event_report_20260208_061726/`
- contains:
  - `REPORT.md`
  - `summary.csv`
  - per-ranked-model files:
    - `*_price_event_alignment.png`
    - `*_test_frame.csv`
    - `*_true_events.csv`
    - `*_pred_events.csv`
    - `*_event_matches.csv`

### Snapshot of event-level result (top baseline model)
For `rank01 dcn_dropout_high` in this run:
- `event_recall_true=0.1667`
- `event_precision_pred=0.3333`
- `mean_abs_lag_days=7.0`

Interpretation note:
- signal remains weak on event timing in this split; chart-based QA confirmed only partial alignment with true TP events.

## 2026-02-08 - Focused hyperparameter grid around the clear DCN winner

### Goal
After identifying a clear winner (`dcn_dropout_high`), run a dedicated DCN-only grid around that neighborhood.

### Focused winner grid setup
Dataset source:
- `/home/rut/ostrofun/RESEARCH/cache/astro_tabular_nn_best_grid__dataset__0faeef66.parquet`

Objective and penalties:
- sort objective: `test_recall_min` desc, `test_recall_gap` asc, `test_mcc` desc, `test_acc` desc
- threshold grid: `0.10 .. 0.90` (`81` points)
- `margin_gap_penalty=0.70`
- `margin_prior_penalty=0.12`

Training envelope:
- epochs: `8`
- seeds: `(44, 45)`
- sampled candidates (`n_trials`): `72`
- effective total runs: `144`

Search space (DCN only):
- `hidden_dims`: `(256,128)`, `(384,192,96)`, `(512,256,128)`, `(640,320,160)`
- `dropout`: `0.25, 0.30, 0.35, 0.40, 0.45`
- `cross_layers`: `2, 3, 4, 5, 6`
- `cross_rank`: `32, 48, 64, 96, 128`
- `learning_rate`: `3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3`
- `weight_decay`: `1e-5, 5e-5, 1e-4, 3e-4, 5e-4`
- `class_weight_power`: `1.2, 1.3, 1.4, 1.5, 1.6`
- `label_smoothing`: `0.00, 0.01, 0.02, 0.05`
- `batch_size`: `384, 512, 768`

### Artifacts
Raw outputs:
- `RESEARCH/reports/astro_tabular_nn_grid_winner_dcn_balance_e8_t72_s2_20260208_062409.csv`
- `RESEARCH/reports/astro_tabular_nn_grid_winner_dcn_balance_e8_t72_s2_20260208_062409.top15.csv`
- `RESEARCH/reports/astro_tabular_nn_grid_winner_dcn_balance_e8_t72_s2_20260208_062409.meta.json`

Notebook-style packaged summary:
- `RESEARCH/reports/astro_tabular_nn_winner_grid_report_20260208_062409/REPORT.md`
- plus copied CSV/TOP15/META in the same folder.

### Outcome vs previous baseline winner
Best row from focused grid:
- run: `54`, seed `45`, `dcn 384x192x96`, dropout `0.45`, cross `3/64`
- test: `recall_min=0.6667`, `recall_gap=0.1302`, `MCC=0.1583`, `ACC=0.7942`

Comparison to baseline `dcn_dropout_high`:
- `recall_min`: `-0.0015` (worse)
- `recall_gap`: `+0.0206` (worse)
- `MCC`: `+0.0269`
- `ACC`: `+0.1239`

Critical selection result:
- runs with `test_recall_min >= baseline(0.6682)`: `0 / 144`

Conclusion:
- baseline winner (`dcn_dropout_high`, seed 45) remains current leader for the target objective.

## 2026-02-08 - Embedded charts directly in markdown reports

### Goal
Make reports easier to review by embedding charts directly inside `REPORT.md` files.

### Code changes
1. Updated price-event report generator:
- `RESEARCH/astro_tabular_nn/price_event_report.py`
- Added `## Charts` section with markdown image links:
  - `![...](rankXX_*_price_event_alignment.png)`
- Kept summary table and added per-rank event metrics next to each embedded chart.

2. Added dedicated winner-grid report generator:
- `RESEARCH/astro_tabular_nn/winner_grid_report.py`
- Generates markdown report plus embedded charts:
  - `chart_recall_vs_gap_scatter.png`
  - `chart_metric_distributions.png`
  - `chart_top_recall_profiles.png`
  - `chart_baseline_vs_best.png`
- Also copies source `csv/top15/meta` into report folder for self-contained review.

3. Documentation update:
- `RESEARCH/astro_tabular_nn/README.md`
- Added both CLIs and explicit note that reports now embed PNG charts.

### Regenerated report artifacts with embedded charts
1. Price-event report markdown updated:
- `RESEARCH/reports/astro_tabular_nn_price_event_report_20260208_061726/REPORT.md`

2. Postrun wide-balance markdown updated with diagnostics images:
- `RESEARCH/reports/astro_tabular_nn_postrun_wide_balance_20260208_055915/REPORT.md`

3. Winner-grid report rebuilt with embedded charts:
- `RESEARCH/reports/astro_tabular_nn_winner_grid_report_20260208_062409/REPORT.md`

## 2026-02-08 - Switch cutoff optimization to TP segment-weighted reward

### Why this change
Observed mismatch: optimizing by `recall_min` alone can still degrade price/segment quality on test markup.
Requested behavior: reward direction correctness more on larger TP segments (old notebook logic).

### What was changed in code
1. Ported old notebook TP segment scoring into module:
- `RESEARCH/astro_tabular_nn/segment_weighted.py`
- Added:
  - `build_tp_segments_from_events(...)`
  - `score_predictions_on_tp_segments(...)`
  - `search_best_threshold_segment_weighted(...)`
  - split-frame helper for `segment_index` datasets.

2. Added objective switch in config:
- `RESEARCH/astro_tabular_nn/config.py`
- New `ScoutConfig` fields:
  - `cutoff_objective` (`recall_balance` | `segment_weighted`)
  - `segment_score_gamma`
  - `segment_min_days`
  - `segment_include_open_tail`
  - `segment_metric`

3. Integrated objective into trainer and outputs:
- `RESEARCH/astro_tabular_nn/trainer.py`
- Validation threshold search now supports segment-weighted objective.
- Split metrics now include:
  - `segment_weighted_hit_rate`
  - `segment_weighted_majority_hit`
  - `cutoff_score`
- Added `cutoff_objective` to fit result schema.

4. Propagated through runners:
- `RESEARCH/astro_tabular_nn/experiments.py`
- `RESEARCH/astro_tabular_nn/grid_search.py`
- `RESEARCH/astro_tabular_nn/quick_scout.py`
- `RESEARCH/astro_tabular_nn/grid_trial.py`
- Runners now pass split frames (`date/close/segment_index/target`) for segment scoring.
- CLI flags added for objective and TP-segment params.

5. Reporting updates:
- `RESEARCH/astro_tabular_nn/postrun_report.py` uses `test_cutoff_score` in sorting when available.
- `RESEARCH/astro_tabular_nn/winner_grid_report.py` supports new ranking field and replay objective auto-detection.
- `RESEARCH/astro_tabular_nn/README.md` updated with new objective examples.

### Validation smoke test
Ran smoke grid:
- command: `python -m RESEARCH.astro_tabular_nn.grid_trial ... --cutoff-objective segment_weighted --n-trials 2 --epochs 1`
- artifact:
  - `RESEARCH/reports/_tmp_smoke_segment_objective.csv`
- confirmed columns produced:
  - `test_cutoff_score`
  - `test_segment_weighted_hit_rate`
  - `test_segment_weighted_majority_hit`

### New directional winner grid under segment-weighted objective
Run setup:
- objective: `segment_weighted`
- score formula (binary): `weighted_hit_rate - 0.70 * recall_gap - 0.12 * prior_gap`
- threshold grid: `0.10 .. 0.90` step `0.01`
- gamma: `1.5`, min segment days: `5`, open tail: `True`
- epochs: `8`
- seeds: `(44, 45)`
- sampled configs: `48` (total runs: `96`)
- search focus: smaller DCN + higher dropout

Artifacts:
- `RESEARCH/reports/astro_tabular_nn_grid_directional_segment_weighted_e8_t48_s2_20260208_070704.csv`
- `RESEARCH/reports/astro_tabular_nn_grid_directional_segment_weighted_e8_t48_s2_20260208_070704.top15.csv`
- `RESEARCH/reports/astro_tabular_nn_grid_directional_segment_weighted_e8_t48_s2_20260208_070704.meta.json`

Top row by `test_cutoff_score`:
- run `26`, seed `44`
- model: `dcn`, hidden `256x128`, dropout `0.60`, cross `6/96`
- test:
  - `test_cutoff_score=0.8465`
  - `weighted_hit_rate=0.9529`
  - `weighted_majority_hit=1.0000`
  - `recall_min=0.6667`
  - `recall_gap=0.1166`

### Notebook-style report package for the new objective
- `RESEARCH/reports/astro_tabular_nn_winner_grid_report_segment_weighted_20260208_070704/REPORT.md`
- includes embedded charts and mandatory test price markup:
  - `chart_price_event_alignment_best.png`
  - plus copied grid/top15/meta and best replay CSV artifacts.
