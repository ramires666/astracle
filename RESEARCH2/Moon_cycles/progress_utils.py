"""Tiny progress-print helpers for long research loops."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _safe_float(row: Dict[str, object], key: str, default: float) -> float:
    """Read a float metric from a dict; treat missing/NaN as default."""
    v = row.get(key, None)
    if v is None:
        return float(default)
    try:
        f = float(v)
    except Exception:
        return float(default)
    return float(default) if np.isnan(f) else f


def _is_better_on_test(candidate: Dict[str, object], best: Optional[Dict[str, object]]) -> bool:
    """
    Decide if candidate is better than best (test-focused comparator).

    Sorting priority (same as our result tables):
    1) maximize test_recall_min
    2) minimize test_recall_gap
    3) maximize test_mcc
    4) maximize test_acc
    """
    if best is None:
        return True

    keys = [
        ("test_recall_min", False),
        ("test_recall_gap", True),
        ("test_mcc", False),
        ("test_acc", False),
    ]

    for k, asc in keys:
        a = _safe_float(candidate, k, default=-1e9 if not asc else 1e9)
        b = _safe_float(best, k, default=-1e9 if not asc else 1e9)
        if a == b:
            continue
        return (a < b) if asc else (a > b)

    return False


def update_best_on_test(
    best: Optional[Dict[str, object]],
    candidate: Dict[str, object],
) -> Dict[str, object]:
    """Return the better of (best, candidate) according to test comparator."""
    return candidate if _is_better_on_test(candidate, best) else (best or candidate)


def _format_cfg(row: Dict[str, object]) -> str:
    """Format identifying config fields (model + gauss params) if present."""
    parts = []
    if "model" in row:
        parts.append(f"model={row.get('model')}")
    if "protocol" in row:
        parts.append(f"protocol={row.get('protocol')}")
    if "gauss_window" in row:
        parts.append(f"gw={int(row.get('gauss_window'))}")
    if "gauss_std" in row:
        parts.append(f"std={float(row.get('gauss_std')):.1f}")
    return " ".join(parts)


def _format_metrics(row: Dict[str, object]) -> str:
    """Format key metrics in a compact, readable one-line style."""
    test = (
        f"test rmin={_safe_float(row,'test_recall_min',np.nan):.3f} "
        f"gap={_safe_float(row,'test_recall_gap',np.nan):.3f} "
        f"mcc={_safe_float(row,'test_mcc',np.nan):.3f} "
        f"acc={_safe_float(row,'test_acc',np.nan):.3f}"
    )

    p = row.get("p_value_vs_random", None)
    if p is not None:
        try:
            test += f" p={float(p):.3g}"
        except Exception:
            pass

    # If validation metrics exist, show them too (helps to see if val/test diverge).
    if "val_recall_min" in row:
        val = (
            f"val rmin={_safe_float(row,'val_recall_min',np.nan):.3f} "
            f"gap={_safe_float(row,'val_recall_gap',np.nan):.3f} "
            f"mcc={_safe_float(row,'val_mcc',np.nan):.3f} "
            f"acc={_safe_float(row,'val_acc',np.nan):.3f}"
        )
        return f"{val} | {test}"

    return test


def progress_update(
    best: Optional[Dict[str, object]],
    current: Dict[str, object],
    done: int,
    total: int,
    prefix: str,
    verbose: bool,
    source: Optional[str] = None,
) -> Dict[str, object]:
    """
    Print a primitive progress line and return updated best.

    We keep this intentionally simple (plain prints) so it works in notebooks,
    terminals, and logs without extra dependencies.
    """
    new_best = update_best_on_test(best, current)

    if not verbose:
        return new_best

    left = max(0, int(total) - int(done))
    cfg = _format_cfg(current)
    met = _format_metrics(current)

    best_cfg = _format_cfg(new_best)
    best_met = _format_metrics(new_best)

    src = f" src={source}" if source else ""

    print(f"[{done}/{total}] left={left} {prefix}{src} {cfg} | {met} | BEST {best_cfg} | {best_met}")

    return new_best
