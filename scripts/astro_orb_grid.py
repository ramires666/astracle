"""
Grid search over orb multipliers for aspects.
Runs astro->features->dataset->split->train and saves a summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.astro_xgb.context import build_context
from src.pipeline.astro_xgb.naming import labels_tag, orb_tag
from src.pipeline.astro_xgb.steps_astro import run_astro_step
from src.pipeline.astro_xgb.steps_features import run_features_step
from src.pipeline.astro_xgb.steps_dataset import run_dataset_step
from src.pipeline.astro_xgb.steps_split import run_split_step
from src.pipeline.astro_xgb.steps_train import run_train_step


def _parse_orb_list(value: str) -> list[float]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [float(p) for p in parts]


def _metrics_path(ctx, dataset_path: Path) -> Path:
    tag = dataset_path.stem.replace(f"{ctx.subject.subject_id}_dataset_", "")
    return ctx.reports_dir / f"xgb_metrics_{tag}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Orb grid search (astro aspects)")
    parser.add_argument("--orb-list", default=None, help="Comma-separated orb multipliers (override config)")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Force recompute steps")
    parser.add_argument(
        "--include-both-centers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute both geo and helio features and merge",
    )
    parser.add_argument("--save-report", action=argparse.BooleanOptionalAction, default=True, help="Save summary CSV/Parquet")
    args = parser.parse_args()

    ctx = build_context(PROJECT_ROOT)

    # Labels tag from config (we assume labels already exist)
    labels_cfg = ctx.cfg_labels.get("labels", {})
    label_tag = labels_tag(
        mode=labels_cfg.get("label_mode", "oracle_gauss"),
        horizon=int(labels_cfg.get("horizon", 1)),
        price_mode=str(labels_cfg.get("price_mode", "raw")),
        gauss_window=int(labels_cfg.get("gauss_window", 201)),
        gauss_std=float(labels_cfg.get("gauss_std", 50.0)),
        move_share=float(labels_cfg.get("target_move_share", 0.5)),
        sigma=float(labels_cfg.get("sigma", 3)),
        threshold=float(labels_cfg.get("threshold", 0.0005)),
        threshold_mode=str(labels_cfg.get("threshold_mode", "auto")),
    )
    labels_path = ctx.processed_dir / f"{ctx.subject.subject_id}_labels_{label_tag}.parquet"
    if not labels_path.exists():
        print("[ERROR] Labels file not found:", labels_path)
        print("Run labels step first:")
        print("  python scripts/astro_pipeline.py --step labels --force")
        return 1

    cfg_astro = ctx.cfg_astro.get("astro", {})
    if args.orb_list:
        orbs = _parse_orb_list(args.orb_list)
    else:
        orbs = [float(v) for v in cfg_astro.get("orb_grid", [0.8, 1.0, 1.2])]
    include_both_centers = bool(args.include_both_centers) if args.include_both_centers is not None else bool(
        cfg_astro.get("include_both_centers", False)
    )
    results = []

    for orb in orbs:
        print("\n" + "=" * 80)
        print(f"[ORB GRID] orb_multiplier = {orb}")
        print("=" * 80)

        astro_paths = run_astro_step(
            ctx,
            market_daily_path=ctx.processed_dir / f"{ctx.subject.subject_id}_market_daily.parquet",
            force=args.force,
            orb_multiplier=orb,
            include_transit_aspects=bool(ctx.cfg_astro.get("astro", {}).get("include_transit_aspects", False)),
            progress=True,
            include_both_centers=include_both_centers,
        )

        features_path = run_features_step(
            ctx,
            astro_paths=astro_paths,
            force=args.force,
            orb_multiplier=orb,
            include_pair_aspects=True,
            include_transit_aspects=bool(ctx.cfg_astro.get("astro", {}).get("include_transit_aspects", False)),
            include_both_centers=include_both_centers,
        )

        dataset_path = run_dataset_step(
            ctx,
            features_path=features_path,
            labels_path=labels_path,
            force=args.force,
            write_inventory=False,
        )

        train_path, val_path, test_path = run_split_step(
            ctx,
            dataset_path=dataset_path,
            force=args.force,
        )

        model_path = run_train_step(
            ctx,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            dataset_path=dataset_path,
            force=args.force,
            threshold_tuning=True,
        )

        metrics_path = _metrics_path(ctx, dataset_path)
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            report = data.get("report", {})
        else:
            metrics = {}
            report = {}

        down = report.get("DOWN", {})
        side = report.get("SIDEWAYS", {})
        up = report.get("UP", {})
        total_support = 0
        if report.get("macro avg", {}).get("support"):
            total_support = report.get("macro avg", {}).get("support")
        results.append({
            "orb_multiplier": orb,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "summary": metrics.get("summary"),
            "bal_acc": metrics.get("bal_acc"),
            "f1_macro": metrics.get("f1_macro"),
            "acc": metrics.get("acc"),
            "down_acc": down.get("recall"),
            "down_f1": down.get("f1-score"),
            "down_freq": (down.get("support", 0) / total_support * 100) if total_support else None,
            "side_acc": side.get("recall"),
            "side_f1": side.get("f1-score"),
            "side_freq": (side.get("support", 0) / total_support * 100) if total_support else None,
            "up_acc": up.get("recall"),
            "up_f1": up.get("f1-score"),
            "up_freq": (up.get("support", 0) / total_support * 100) if total_support else None,
        })

    df = pd.DataFrame(results).sort_values(["f1_macro", "bal_acc"], ascending=False)
    print("\n[ORB GRID] Summary (top 10):")
    print(df.head(10))

    if args.save_report:
        out_csv = ctx.reports_dir / "orb_grid_summary.csv"
        out_parquet = ctx.reports_dir / "orb_grid_summary.parquet"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        df.to_parquet(out_parquet, index=False)
        print("[ORB GRID] Saved:", out_csv)
        print("[ORB GRID] Saved:", out_parquet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
