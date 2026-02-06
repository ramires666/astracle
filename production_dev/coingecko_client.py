"""
CoinGecko API client helpers for production service.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import requests


PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _read_env_value(name: str, env_path: Path = ENV_PATH) -> Optional[str]:
    """
    Read a single key from .env without external dependencies.
    """
    if not env_path.exists():
        return None

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            if key.strip() != name:
                continue
            val = value.strip().strip('"').strip("'")
            return val or None
    except Exception:
        return None
    return None


def get_coingecko_api_key() -> Optional[str]:
    """
    Resolve CoinGecko API key:
    1) process env COINGECKO
    2) .env COINGECKO in project root
    """
    import os

    key = os.getenv("COINGECKO")
    if key:
        return key.strip()
    return _read_env_value("COINGECKO")


def fetch_current_btc_price_usd(timeout: int = 10) -> float:
    """
    Fetch current BTC price in USD from CoinGecko.
    Uses COINGECKO API key when available.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    key = get_coingecko_api_key()

    headers = {}
    if key:
        headers["x-cg-pro-api-key"] = key

    response = requests.get(url, params=params, headers=headers, timeout=timeout)

    # If key format is not accepted as PRO, retry DEMO header once.
    if response.status_code in (401, 403) and key:
        demo_headers = {"x-cg-demo-api-key": key}
        response = requests.get(url, params=params, headers=demo_headers, timeout=timeout)

    response.raise_for_status()
    data = response.json()
    return float(data["bitcoin"]["usd"])


def fetch_btc_daily_prices_usd(
    days: Union[int, str] = 120,
    timeout: int = 20,
) -> list[tuple[str, float]]:
    """
    Fetch BTC daily prices in USD from CoinGecko market_chart endpoint.

    Args:
        days: Number of days or "max".
        timeout: HTTP timeout in seconds.

    Returns:
        Sorted list of (YYYY-MM-DD, close_price) tuples.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": str(days),
        "interval": "daily",
    }
    key = get_coingecko_api_key()

    headers = {}
    if key:
        headers["x-cg-pro-api-key"] = key

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    if response.status_code in (401, 403) and key:
        demo_headers = {"x-cg-demo-api-key": key}
        response = requests.get(url, params=params, headers=demo_headers, timeout=timeout)

    response.raise_for_status()
    payload = response.json()
    raw_prices = payload.get("prices", [])

    by_date: dict[str, float] = {}
    for item in raw_prices:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts_ms, px = item[0], item[1]
        try:
            dt = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).date().isoformat()
            by_date[dt] = float(px)
        except Exception:
            continue

    return sorted(by_date.items(), key=lambda x: x[0])
