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
