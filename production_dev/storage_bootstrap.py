"""
Runtime storage bootstrap helpers.

Goal:
- If deploy bind-mount folders are empty on first run, seed them from
  snapshot files baked into the image under /app/bootstrap/data.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).parent.parent

RUNTIME_MARKET_DIR = PROJECT_ROOT / "data" / "market"
RUNTIME_PREDICTION_CACHE_DIR = PROJECT_ROOT / "data" / "prediction_cache"

SEED_MARKET_DIR = PROJECT_ROOT / "bootstrap" / "data" / "market"
SEED_PREDICTION_CACHE_DIR = PROJECT_ROOT / "bootstrap" / "data" / "prediction_cache"


def _copy_missing_tree(seed_dir: Path, target_dir: Path) -> int:
    """
    Copy files from seed_dir to target_dir only when destination files are missing.
    """
    if not seed_dir.exists():
        return 0

    copied = 0
    for src in seed_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(seed_dir)
        dst = target_dir / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _has_market_snapshot(market_dir: Path) -> bool:
    """
    Minimal marker that local market history exists.
    """
    parquet_path = market_dir / "processed" / "BTC_full_market_daily.parquet"
    csv_path = market_dir / "processed" / "BTC_full_market_daily.csv"
    return parquet_path.exists() or csv_path.exists()


def _has_prediction_cache(cache_dir: Path) -> bool:
    """
    Minimal marker that cache files exist.
    """
    forecast_path = cache_dir / "forecast_predictions.parquet"
    backtest_path = cache_dir / "backtest_predictions.parquet"
    return forecast_path.exists() or backtest_path.exists()


def ensure_runtime_storage_seed(verbose: bool = True) -> Dict[str, object]:
    """
    Seed empty runtime folders from image snapshots.
    """
    RUNTIME_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    result: Dict[str, object] = {
        "market_seeded": False,
        "market_files_copied": 0,
        "prediction_cache_seeded": False,
        "prediction_cache_files_copied": 0,
    }

    if not _has_market_snapshot(RUNTIME_MARKET_DIR):
        copied = _copy_missing_tree(SEED_MARKET_DIR, RUNTIME_MARKET_DIR)
        if copied > 0 or _has_market_snapshot(RUNTIME_MARKET_DIR):
            result["market_seeded"] = True
            result["market_files_copied"] = int(copied)
            if verbose:
                print(
                    f"[STORAGE-BOOTSTRAP] Seeded market snapshot files: {copied} "
                    f"(target={RUNTIME_MARKET_DIR})"
                )

    if not _has_prediction_cache(RUNTIME_PREDICTION_CACHE_DIR):
        copied = _copy_missing_tree(SEED_PREDICTION_CACHE_DIR, RUNTIME_PREDICTION_CACHE_DIR)
        if copied > 0 or _has_prediction_cache(RUNTIME_PREDICTION_CACHE_DIR):
            result["prediction_cache_seeded"] = True
            result["prediction_cache_files_copied"] = int(copied)
            if verbose:
                print(
                    f"[STORAGE-BOOTSTRAP] Seeded prediction cache files: {copied} "
                    f"(target={RUNTIME_PREDICTION_CACHE_DIR})"
                )

    return result
