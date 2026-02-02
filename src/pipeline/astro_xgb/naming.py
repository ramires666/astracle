"""
Filename helpers for astro_xgb pipeline artifacts.
"""

from __future__ import annotations

from pathlib import Path


def _float_tag(value: float, digits: int = 2) -> str:
    """
    Convert float to a filesystem-friendly tag.
    """
    fmt = f"{value:.{digits}f}"
    return fmt.replace(".", "p")


def orb_tag(orb_mult: float) -> str:
    return f"om{_float_tag(float(orb_mult), digits=2)}"


def gauss_tag(window: int, std: float) -> str:
    return f"gw{int(window)}_gs{_float_tag(float(std), digits=1)}"


def labels_tag(
    mode: str,
    horizon: int,
    price_mode: str,
    gauss_window: int,
    gauss_std: float,
    move_share: float,
    sigma: float | None = None,
    threshold: float | None = None,
    threshold_mode: str | None = None,
) -> str:
    if str(mode).lower().startswith("oracle"):
        s = sigma if sigma is not None else 3.0
        t = threshold if threshold is not None else 0.0005
        tm = (threshold_mode or "fixed").lower()
        return (
            f"{mode}_h{int(horizon)}_pm{price_mode}_"
            f"s{_float_tag(float(s), digits=1)}_thr{_float_tag(float(t), digits=6)}_tm{tm}"
        )
    return (
        f"{mode}_h{int(horizon)}_pm{price_mode}_"
        f"{gauss_tag(gauss_window, gauss_std)}_ms{_float_tag(float(move_share), digits=2)}"
    )


def with_suffix(path: Path, suffix: str) -> Path:
    """
    Append a suffix before file extension.
    """
    path = Path(path)
    return path.with_name(path.stem + suffix + path.suffix)
