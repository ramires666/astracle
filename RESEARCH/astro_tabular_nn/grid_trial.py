"""CLI entrypoint for broad grid-search trial on astro tabular NN."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .best_grid_dataset import ensure_best_grid_dataset_path
from .config import DatasetConfig, ScoutConfig, TrainConfig, with_batch_size, with_dataset, with_epochs
from .data_utils import load_tabular_dataset
from .grid_search import GridSearchSpace, run_broad_grid_trial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run broad trial grid-search for astro tabular NN")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to parquet dataset (optional; overrides dataset-source)",
    )
    p.add_argument(
        "--dataset-source",
        type=str,
        choices=["best-grid", "default-parquet"],
        default="best-grid",
        help="How to pick dataset when --dataset is not provided",
    )
    p.add_argument("--run-tag", type=str, default="turning_massive_label_grid", help="Checkpoint run tag")
    p.add_argument("--data-start", type=str, default="2017-11-01", help="Market start date")
    p.add_argument("--epochs", type=int, default=6, help="Epochs per run")
    p.add_argument("--batch-size", type=int, default=512, help="Base batch size in TrainConfig")
    p.add_argument("--n-trials", type=int, default=18, help="Number of sampled candidates")
    p.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=["dcn", "deepfm"],
        help="Architectures to include in candidate pool",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Training seeds",
    )
    p.add_argument("--sample-seed", type=int, default=1729, help="Seed for candidate sampling")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV for grid results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset:
        dataset_path = Path(args.dataset)
    elif args.dataset_source == "best-grid":
        dataset_path = ensure_best_grid_dataset_path(
            run_tag=str(args.run_tag),
            data_start=str(args.data_start),
            use_cache=True,
            verbose=True,
        )
    else:
        dataset_path = DatasetConfig().dataset_path

    ds_cfg = with_dataset(DatasetConfig(), dataset_path)
    train_cfg = with_epochs(TrainConfig(), int(args.epochs))
    train_cfg = with_batch_size(train_cfg, int(args.batch_size))
    scout_cfg = ScoutConfig(train=train_cfg)
    model_types = tuple(str(v).lower().strip() for v in args.model_types)
    space = replace(GridSearchSpace(), model_types=model_types)

    print("[grid_trial] loading dataset:", ds_cfg.dataset_path)
    dataset = load_tabular_dataset(ds_cfg)

    results, meta = run_broad_grid_trial(
        dataset=dataset,
        scout_cfg=scout_cfg,
        space=space,
        n_trials=int(args.n_trials),
        seeds=tuple(int(s) for s in args.seeds),
        sample_seed=int(args.sample_seed),
        verbose=True,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)

    print("\n[grid_trial] split summary")
    print(pd.Series(meta["split_summary"]).to_string())

    print("\n[grid_trial] top results")
    cols = [
        "run_id",
        "seed",
        "model_type",
        "hidden_dims",
        "dropout",
        "cross_layers",
        "cross_rank",
        "embed_dim",
        "learning_rate",
        "weight_decay",
        "class_weight_power",
        "label_smoothing",
        "batch_size",
        "cutoff_kind",
        "best_margin",
        "best_val_score",
        "test_recall_min",
        "test_recall_gap",
        "test_mcc",
        "test_acc",
    ]
    print(results[cols].head(12).to_string(index=False))
    print("\n[grid_trial] saved:", out_csv)


if __name__ == "__main__":
    main()
