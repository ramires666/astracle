# Astro Tabular NN (New Direction)

This folder contains a new CUDA-first neural research direction for astro-only tabular features.

## Goals

- Keep all reusable logic in Python modules (usable from scripts and notebooks).
- Run the first scout experiments in notebook mode to estimate signal potential.
- Use vectorized preprocessing, Numba where useful, and mandatory CUDA training.

## Structure

- `config.py`: dataclass configs for dataset, split, model, and training.
- `data_utils.py`: parquet loading, chronological split, robust scaling, class weights.
- `best_grid_dataset.py`: rebuilds exact `grid_best` labeled dataset from latest TP notebook checkpoint.
- `model_dcn.py`: low-rank cross network + residual MLP architecture.
- `model_deepfm.py`: DeepFM-style interaction network for tabular astro signals.
- `metrics_numba.py`: Numba-accelerated margin scan and metrics helpers.
- `trainer.py`: CUDA AMP training loop with early stopping.
- `experiments.py`: quick scout run orchestration and rough tuning bound suggestions.
- `quick_scout.py`: CLI entrypoint for short multi-run scouting.
- `grid_search.py`: broad trial search over architecture + training hyperparameters.
- `grid_trial.py`: CLI entrypoint for sampled broad grid-search trial.
- `postrun_report.py`: default notebook post-run visual diagnostics (confusion matrix + class balance + up/down recall).
- `astro_tabular_nn_quick_scout.ipynb`: notebook for initial exploratory runs.
- `astro_tabular_nn_grid_trial.ipynb`: notebook for sampled broad grid trial.
- `research-history.md`: chronological log of research steps and reproducibility notes.

## Quick CLI Usage

```bash
python -m RESEARCH.astro_tabular_nn.quick_scout \
  --dataset-source best-grid \
  --run-tag turning_massive_label_grid \
  --data-start 2017-11-01 \
  --epochs 8 \
  --batch-size 512 \
  --seeds 42 43 \
  --out-csv RESEARCH/reports/astro_tabular_nn_quick_scout.csv
```

```bash
python -m RESEARCH.astro_tabular_nn.grid_trial \
  --dataset-source best-grid \
  --run-tag turning_massive_label_grid \
  --data-start 2017-11-01 \
  --epochs 6 \
  --n-trials 24 \
  --model-types dcn deepfm \
  --seeds 42 \
  --sample-seed 1729 \
  --out-csv RESEARCH/reports/astro_tabular_nn_grid_trial.csv
```

## Notes

- CUDA is required by this direction. Training will raise an error on CPU-only runtime.
- Scout runs are intentionally short and are not final model selection.
- Suggested bounds from scout output are only a starting point for deeper tuning.
- `grid_trial` now supports two architectures: `dcn` and `deepfm`.
- For `best-grid` dataset, target is binary and trainer uses threshold search (not ternary margin).
- `next_ret` and target-side helper columns are excluded from features to avoid leakage.
- Notebooks now run post-run diagnostics by default immediately after each trial/scout run.
