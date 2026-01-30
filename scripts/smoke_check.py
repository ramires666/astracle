"""
Simple environment smoke test.
Checks imports of key libraries and prints versions.
"""

from __future__ import annotations

import importlib
import platform
import sys


def _read_version(module, custom_getter: str | None) -> str:
    """
    Try to read module version safely.
    """
    if custom_getter and hasattr(module, custom_getter):
        getter = getattr(module, custom_getter)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                return "unknown"
        return str(getter)

    # Standard options: __version__, VERSION, version
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(module, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                return str(value())
            except Exception:
                return "unknown"
        return str(value)

    return "unknown"


def _check_import(name: str, custom_getter: str | None = None) -> tuple[bool, str]:
    """
    Return (success, version/error) for the given module.
    """
    try:
        module = importlib.import_module(name)
        version = _read_version(module, custom_getter)
        return True, version
    except Exception as exc:  # noqa: BLE001 - keep the real import error
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    """
    Entry point: print info and exit code.
    """
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print("")

    # Key project dependencies
    checks: list[tuple[str, str | None]] = [
        ("fastapi", None),
        ("uvicorn", None),
        ("pydantic", None),
        ("pandas", None),
        ("svgwrite", None),
        ("swisseph", "swe_version"),
    ]

    has_errors = False
    for name, getter in checks:
        ok, info = _check_import(name, getter)
        status = "OK" if ok else "FAIL"
        print(f"{status:>4} | {name:<10} | {info}")
        if not ok:
            has_errors = True

    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
