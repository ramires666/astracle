"""
CLI pipeline for astro_xgboost_training_balanced notebook.
Run steps individually or in ranges with caching.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.astro_xgb.context import build_context
from src.pipeline.astro_xgb.naming import labels_tag, orb_tag
from src.pipeline.astro_xgb.steps_market import run_market_step
from src.pipeline.astro_xgb.steps_labels import run_labels_step
from src.pipeline.astro_xgb.steps_astro import run_astro_step
from src.pipeline.astro_xgb.steps_features import run_features_step
from src.pipeline.astro_xgb.steps_dataset import run_dataset_step
from src.pipeline.astro_xgb.steps_split import run_split_step, _split_paths
from src.pipeline.astro_xgb.steps_train import run_train_step
from src.pipeline.astro_xgb.steps_eval import run_eval_step


STEP_ORDER = ["market", "labels", "astro", "features", "dataset", "split", "train", "eval"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astro XGBoost pipeline (balanced labels)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", choices=STEP_ORDER, help="Run a single step only")
    group.add_argument("--from-step", choices=STEP_ORDER, help="Start from a specific step")
    parser.add_argument("--to-step", choices=STEP_ORDER, help="Stop at a specific step (inclusive)")

    parser.add_argument("--subject-id", default=None, help="Override subject_id from configs/subjects.yaml")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Ignore cache")
    parser.add_argument("--market-update", action=argparse.BooleanOptionalAction, default=False, help="Download/update market data")
    parser.add_argument(
        "--save-db",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write market_daily into DB (default: market.write_db or true)",
    )

    parser.add_argument("--data-start", default="2017-11-01", help="Filter market data start (YYYY-MM-DD)")

    # Label params
    parser.add_argument("--label-mode", default=None, help="balanced_future_return or balanced_detrended")
    parser.add_argument("--label-price-mode", default=None, help="raw or log")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon (days)")
    parser.add_argument("--gauss-window", type=int, default=None, help="Gaussian window (odd)")
    parser.add_argument("--gauss-std", type=float, default=None, help="Gaussian std")
    parser.add_argument("--target-move-share", type=float, default=None, help="Target move share (0..1)")
    parser.add_argument("--sigma", type=float, default=None, help="Oracle sigma for Gaussian smoothing")
    parser.add_argument("--threshold", type=float, default=None, help="Oracle slope threshold")
    parser.add_argument("--threshold-mode", default=None, help="Oracle threshold mode: auto|fixed")
    parser.add_argument("--grid-search", action=argparse.BooleanOptionalAction, default=False, help="Run label grid search")
    parser.add_argument("--grid-threshold-mode", default=None, help="Grid threshold mode: range|quantile")
    parser.add_argument("--grid-sigma-min", type=float, default=None, help="Grid sigma min")
    parser.add_argument("--grid-sigma-max", type=float, default=None, help="Grid sigma max")
    parser.add_argument("--grid-threshold-min", type=float, default=None, help="Grid threshold min")
    parser.add_argument("--grid-threshold-max", type=float, default=None, help="Grid threshold max")
    parser.add_argument("--grid-steps", type=int, default=None, help="Grid steps")
    parser.add_argument("--grid-quantile-min", type=float, default=None, help="Grid quantile min")
    parser.add_argument("--grid-quantile-max", type=float, default=None, help="Grid quantile max")
    parser.add_argument("--balanced-grid-search", action=argparse.BooleanOptionalAction, default=False, help="Balanced labels grid search")
    parser.add_argument("--balanced-grid-apply-best", action=argparse.BooleanOptionalAction, default=False, help="Apply best balanced grid params")
    parser.add_argument("--balanced-grid-windows", default=None, help="Comma list for gauss_window grid")
    parser.add_argument("--balanced-grid-stds", default=None, help="Comma list for gauss_std grid")
    parser.add_argument("--balanced-grid-move-shares", default=None, help="Comma list for target_move_share grid")
    parser.add_argument(
        "--balanced-edge-fill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fill Gaussian smoothing edges for balanced_detrended labels",
    )
    parser.add_argument(
        "--open-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open saved plots in the default viewer",
    )

    # Astro params
    parser.add_argument("--orb-multiplier", type=float, default=None, help="Orb multiplier for aspects")
    parser.add_argument(
        "--include-transit-aspects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable transit->natal aspects",
    )
    parser.add_argument(
        "--include-both-centers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute both geo and helio features and merge",
    )

    parser.add_argument("--write-inventory", action=argparse.BooleanOptionalAction, default=False, help="Save feature inventory")
    parser.add_argument(
        "--write-label-report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write dxcharts label report (defaults to labels.write_report)",
    )
    parser.add_argument(
        "--label-report-sample-days",
        type=int,
        default=None,
        help="Downsample labels report: 1 point per N days (defaults to labels.report_sample_every_days)",
    )
    parser.add_argument(
        "--label-report-max-points",
        type=int,
        default=None,
        help="Cap labels report data points (defaults to labels.report_max_points)",
    )
    parser.add_argument(
        "--threshold-tuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune threshold for binary models",
    )

    return parser.parse_args()


def _steps_to_run(args: argparse.Namespace) -> list[str]:
    if args.step:
        return [args.step]
    if args.from_step:
        start_idx = STEP_ORDER.index(args.from_step)
    else:
        start_idx = 0
    if args.to_step:
        end_idx = STEP_ORDER.index(args.to_step)
    else:
        end_idx = len(STEP_ORDER) - 1
    return STEP_ORDER[start_idx : end_idx + 1]


def main() -> int:
    args = _parse_args()
    steps = _steps_to_run(args)

    ctx = build_context(PROJECT_ROOT, subject_id=args.subject_id)

    # Precompute tags to infer expected paths when running partial steps.
    label_tag = labels_tag(
        mode=args.label_mode or ctx.cfg_labels.get("labels", {}).get("label_mode", "balanced_detrended"),
        horizon=int(args.horizon or ctx.cfg_labels.get("labels", {}).get("horizon", 1)),
        price_mode=str(args.label_price_mode or ctx.cfg_labels.get("labels", {}).get("price_mode", "raw")),
        gauss_window=int(args.gauss_window or ctx.cfg_labels.get("labels", {}).get("gauss_window", 201)),
        gauss_std=float(args.gauss_std or ctx.cfg_labels.get("labels", {}).get("gauss_std", 50.0)),
        move_share=float(args.target_move_share or ctx.cfg_labels.get("labels", {}).get("target_move_share", 0.5)),
        sigma=float(args.sigma or ctx.cfg_labels.get("labels", {}).get("sigma", 3)),
        threshold=float(args.threshold or ctx.cfg_labels.get("labels", {}).get("threshold", 0.0005)),
        threshold_mode=str(args.threshold_mode or ctx.cfg_labels.get("labels", {}).get("threshold_mode", "auto")),
    )
    cfg_astro = ctx.cfg_astro.get("astro", {})
    orb_multiplier = float(args.orb_multiplier if args.orb_multiplier is not None else cfg_astro.get("orb_multiplier", 1.0))
    include_both_centers = bool(args.include_both_centers) if args.include_both_centers is not None else bool(
        cfg_astro.get("include_both_centers", False)
    )
    include_transit_aspects = bool(args.include_transit_aspects) if args.include_transit_aspects is not None else bool(
        cfg_astro.get("include_transit_aspects", False)
    )
    orb = orb_tag(orb_multiplier)

    market_path = ctx.processed_dir / f"{ctx.subject.subject_id}_market_daily.parquet"
    labels_path = ctx.processed_dir / f"{ctx.subject.subject_id}_labels_{label_tag}.parquet"
    cfg_center = cfg_astro.get("center", "geo")
    centers_tag = "both" if include_both_centers else cfg_center
    features_path = ctx.processed_dir / f"{ctx.subject.subject_id}_features_{orb}_{centers_tag}.parquet"
    dataset_path = ctx.processed_dir / f"{ctx.subject.subject_id}_dataset_{label_tag}__{orb}_{centers_tag}.parquet"
    if include_both_centers:
        astro_paths = {
            "geo": (
                ctx.processed_dir / f"{ctx.subject.subject_id}_astro_bodies_geo.parquet",
                ctx.processed_dir / f"{ctx.subject.subject_id}_astro_aspects_{orb}_geo.parquet",
                ctx.processed_dir / f"{ctx.subject.subject_id}_transit_aspects_{orb}_geo.parquet",
            ),
            "helio": (
                ctx.processed_dir / f"{ctx.subject.subject_id}_astro_bodies_helio.parquet",
                ctx.processed_dir / f"{ctx.subject.subject_id}_astro_aspects_{orb}_helio.parquet",
                ctx.processed_dir / f"{ctx.subject.subject_id}_transit_aspects_{orb}_helio.parquet",
            ),
        }
    else:
        astro_paths = {
            cfg_center: (
                ctx.processed_dir / f"{ctx.subject.subject_id}_astro_bodies.parquet",
                ctx.processed_dir / f"{ctx.subject.subject_id}_astro_aspects_{orb}.parquet",
                ctx.processed_dir / f"{ctx.subject.subject_id}_transit_aspects_{orb}.parquet",
            )
        }

    train_path, val_path, test_path = _split_paths(ctx, dataset_path)
    model_path = ctx.root / "models_artifacts" / f"xgb_astro_{dataset_path.stem.replace(ctx.subject.subject_id + '_dataset_', '')}.joblib"

    # Helpful pre-check: if we need labels but they're missing, list available ones.
    if "labels" not in steps and any(s in steps for s in ["dataset", "split", "train", "eval"]):
        if not labels_path.exists():
            available = sorted(ctx.processed_dir.glob(f"{ctx.subject.subject_id}_labels_*.parquet"))
            print("[ERROR] Expected labels file not found:", labels_path)
            print("[ERROR] Config labels params:")
            print("  label_mode     =", ctx.cfg_labels.get("labels", {}).get("label_mode"))
            print("  sigma          =", ctx.cfg_labels.get("labels", {}).get("sigma"))
            print("  threshold      =", ctx.cfg_labels.get("labels", {}).get("threshold"))
            print("  threshold_mode =", ctx.cfg_labels.get("labels", {}).get("threshold_mode"))
            print("  price_mode     =", ctx.cfg_labels.get("labels", {}).get("price_mode"))
            print("  horizon        =", ctx.cfg_labels.get("labels", {}).get("horizon"))
            if available:
                print("[ERROR] Available labels files:")
                for p in available:
                    print("  ", p.name)
            print("[ERROR] Fix by:")
            print("  - updating configs/labels.yaml (sigma/threshold/threshold_mode), or")
            print("  - running labels step with explicit params, e.g.:")
            print("    python scripts/astro_pipeline.py --step labels --sigma 4 --threshold 0.0006 --threshold-mode fixed --force")
            raise FileNotFoundError(labels_path)

    if "market" in steps:
        market_path = run_market_step(
            ctx,
            force=args.force,
            update=args.market_update,
            prefer_binance=True,
            data_start=args.data_start,
            save_db=args.save_db,
        )

    if "labels" in steps:
        labels_path = run_labels_step(
            ctx,
            market_daily_path=market_path,
            force=args.force,
            label_mode=args.label_mode,
            price_mode=args.label_price_mode,
            horizon=args.horizon,
            gauss_window=args.gauss_window,
            gauss_std=args.gauss_std,
            target_move_share=args.target_move_share,
            sigma=args.sigma,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            grid_search=args.grid_search,
            grid_threshold_mode=args.grid_threshold_mode,
            grid_sigma_min=args.grid_sigma_min,
            grid_sigma_max=args.grid_sigma_max,
            grid_threshold_min=args.grid_threshold_min,
            grid_threshold_max=args.grid_threshold_max,
            grid_steps=args.grid_steps,
            grid_quantile_min=args.grid_quantile_min,
            grid_quantile_max=args.grid_quantile_max,
            balanced_grid_search=args.balanced_grid_search,
            balanced_grid_apply_best=args.balanced_grid_apply_best,
            balanced_grid_windows=args.balanced_grid_windows,
            balanced_grid_stds=args.balanced_grid_stds,
            balanced_grid_move_shares=args.balanced_grid_move_shares,
            write_report=args.write_label_report,
            open_plots=args.open_plots,
            balanced_edge_fill=args.balanced_edge_fill,
            report_sample_every_days=args.label_report_sample_days,
            report_max_points=args.label_report_max_points,
        )

    if "astro" in steps:
        astro_paths = run_astro_step(
            ctx,
            market_daily_path=market_path,
            force=args.force,
            orb_multiplier=orb_multiplier,
            include_transit_aspects=include_transit_aspects,
            progress=True,
            include_both_centers=include_both_centers,
        )

    if "features" in steps:
        features_path = run_features_step(
            ctx,
            astro_paths=astro_paths,
            force=args.force,
            orb_multiplier=orb_multiplier,
            include_pair_aspects=True,
            include_transit_aspects=include_transit_aspects,
            include_both_centers=include_both_centers,
        )

    if "dataset" in steps:
        dataset_path = run_dataset_step(
            ctx,
            features_path=features_path,
            labels_path=labels_path,
            force=args.force,
            write_inventory=args.write_inventory,
        )

    if "split" in steps:
        train_path, val_path, test_path = run_split_step(
            ctx,
            dataset_path=dataset_path,
            force=args.force,
        )

    if "train" in steps:
        model_path = run_train_step(
            ctx,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            dataset_path=dataset_path,
            force=args.force,
            threshold_tuning=bool(args.threshold_tuning),
        )

    if "eval" in steps:
        run_eval_step(
            ctx,
            model_path=model_path,
            test_path=test_path,
            force=args.force,
        )

    print("[OK] Steps completed:", ", ".join(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
