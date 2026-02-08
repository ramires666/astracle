"""CLI entrypoint for fast CUDA scout runs of tabular astro neural model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .best_grid_dataset import ensure_best_grid_dataset_path
from .config import DatasetConfig, ScoutConfig, TrainConfig, with_dataset, with_epochs, with_batch_size
from .data_utils import load_tabular_dataset
from .experiments import default_scout_model_grid, run_quick_scout, save_scout_results, suggest_tuning_bounds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quick scout experiments for astro tabular NN")
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
    p.add_argument("--epochs", type=int, default=12, help="Epochs per run for scout mode")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43],
        help="Random seeds for repeated short runs",
    )
    p.add_argument(
        "--cutoff-objective",
        type=str,
        choices=["recall_balance", "segment_weighted"],
        default="recall_balance",
        help="Threshold objective: classic recall balance or TP-segment weighted reward",
    )
    p.add_argument("--segment-score-gamma", type=float, default=1.5, help="TP segment amplitude exponent gamma")
    p.add_argument("--segment-min-days", type=int, default=5, help="Minimum TP segment days")
    p.add_argument(
        "--segment-no-open-tail",
        action="store_true",
        help="Disable open-tail segment in TP segment-weighted objective",
    )
    p.add_argument("--out-csv", type=str, default=None, help="Optional output CSV")
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

    scout_cfg = ScoutConfig(
        train=train_cfg,
        cutoff_objective=str(args.cutoff_objective),
        segment_score_gamma=float(args.segment_score_gamma),
        segment_min_days=int(args.segment_min_days),
        segment_include_open_tail=not bool(args.segment_no_open_tail),
    )

    print("[quick_scout] loading dataset:", ds_cfg.dataset_path)
    data = load_tabular_dataset(ds_cfg)

    specs = default_scout_model_grid()
    results, meta = run_quick_scout(
        dataset=data,
        scout_cfg=scout_cfg,
        model_specs=specs,
        seeds=tuple(int(s) for s in args.seeds),
        verbose=True,
    )

    print("\n[quick_scout] split summary")
    print(pd.Series(meta["split_summary"]).to_string())

    print("\n[quick_scout] top results")
    cols = [
        "model",
        "model_type",
        "seed",
        "cutoff_kind",
        "cutoff_objective",
        "best_margin",
        "best_val_score",
        "test_cutoff_score",
        "test_segment_weighted_hit_rate",
        "test_segment_weighted_majority_hit",
        "test_recall_min",
        "test_recall_gap",
        "test_mcc",
        "test_acc",
    ]
    cols = [c for c in cols if c in results.columns]
    print(results[cols].head(10).to_string(index=False))

    bounds = suggest_tuning_bounds(results, top_k=3)
    print("\n[quick_scout] suggested bounds")
    print(pd.Series(bounds).to_string())

    save_scout_results(results, args.out_csv)
    if args.out_csv:
        print("\n[quick_scout] saved:", args.out_csv)


if __name__ == "__main__":
    main()
