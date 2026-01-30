"""
Download daily (1d) data from the public Binance Vision archive.

For MVP we only need daily data, so this is a simplified downloader.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests


# Base URLs for Binance Vision
BASE_URL_FUTURES_UM = "https://data.binance.vision/data/futures/um/monthly"
BASE_URL_SPOT = "https://data.binance.vision/data/spot/monthly"


def _iter_with_progress(items: Iterable, enabled: bool, desc: str):
    """
    Progress wrapper. If tqdm is not installed, return items as-is.
    """
    if not enabled:
        return items

    try:
        from tqdm import tqdm
    except ImportError:
        print("[WARN] tqdm is not installed, progress disabled.")
        return items

    return tqdm(items, desc=desc, unit="month")


def _get_base_url(market_type: str) -> str:
    """
    Pick base URL by market type.
    """
    if market_type == "spot":
        return BASE_URL_SPOT
    return BASE_URL_FUTURES_UM


def generate_monthly_dates(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Tuple[int, str]]:
    """
    Generate list of (year, month) from start to end inclusive.
    """
    dates: List[Tuple[int, str]] = []
    cur = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)

    while cur <= end:
        dates.append((cur.year, f"{cur.month:02d}"))
        # Move to next month
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    return dates


def _check_exists(url: str) -> bool:
    """
    Check if file exists by URL.
    Use HEAD; if it fails, fallback to GET with stream.
    """
    try:
        r = requests.head(url, timeout=15)
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
    except requests.RequestException:
        pass

    # Fallback via GET
    try:
        r = requests.get(url, stream=True, timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False


def find_first_available_month(
    symbol: str,
    interval: str,
    market_type: str,
    start_year: int = 2010,
    start_month: int = 1,
    verbose: bool = True,
) -> Optional[Tuple[int, int]]:
    """
    Find the first available month in the archive.
    Scan forward from the start year until a file exists.
    """
    now = datetime.utcnow()
    base_url = _get_base_url(market_type)

    if verbose:
        print(f"[AUTO] Searching first month for {symbol} ({market_type}, {interval})")

    for year in range(start_year, now.year + 1):
        for month in range(1, 13):
            # Do not check future months
            if year == now.year and month > now.month:
                break

            mm = f"{month:02d}"
            filename = f"{symbol}-{interval}-{year}-{mm}.zip"
            url = f"{base_url}/klines/{symbol}/{interval}/{filename}"

            if _check_exists(url):
                if verbose:
                    print(f"[AUTO] Found first month: {year}-{mm}")
                return year, month

    return None


def download_file(
    url: str,
    destination_path: Path,
    verbose: bool = True,
    print_urls: bool = False,
) -> bool:
    """
    Download a file by URL to destination_path.
    """
    try:
        if verbose and print_urls:
            print(f"[GET] {url}")

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if verbose:
            print(f"[OK] {destination_path.name}")
        return True
    except requests.RequestException as e:
        if verbose:
            if "404" in str(e):
                print(f"[404] {destination_path.name}")
            else:
                print(f"[ERR] {destination_path.name}: {e}")
        return False


def download_monthly_1d(
    symbol: str,
    market_type: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    dest_folder: Path,
    progress: bool = False,
    verbose: bool = True,
    print_urls: bool = False,
) -> List[Path]:
    """
    Download all monthly 1d archives for the given period.
    """
    base_url = _get_base_url(market_type)
    interval = "1d"

    dest_folder.mkdir(parents=True, exist_ok=True)
    dates = generate_monthly_dates(start_year, start_month, end_year, end_month)

    if verbose:
        print(
            f"[INFO] Download {symbol} {interval} ({market_type}) "
            f"from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}"
        )
        print(f"[INFO] Destination folder: {dest_folder}")
        print(f"[INFO] Months count: {len(dates)}")

    downloaded: List[Path] = []
    for year, month in _iter_with_progress(dates, progress, desc="download 1d"):
        filename = f"{symbol}-{interval}-{year}-{month}.zip"
        url = f"{base_url}/klines/{symbol}/{interval}/{filename}"
        dest_path = dest_folder / filename

        if dest_path.exists():
            if verbose:
                print(f"[SKIP] {filename}")
            downloaded.append(dest_path)
            continue

        if download_file(url, dest_path, verbose=verbose, print_urls=print_urls):
            downloaded.append(dest_path)

    return downloaded


def download_1d_auto(
    symbol: str,
    market_type: str,
    data_root: Path,
    auto_start: bool = True,
    start_year: int = 2010,
    start_month: int = 1,
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    progress: bool = False,
    verbose: bool = True,
    print_urls: bool = False,
) -> List[Path]:
    """
    Smart mode: find the earliest available month and download up to current.
    """
    now = datetime.utcnow()
    end_year = end_year or now.year
    end_month = end_month or now.month

    if auto_start:
        found = find_first_available_month(
            symbol=symbol,
            interval="1d",
            market_type=market_type,
            start_year=start_year,
            start_month=start_month,
            verbose=verbose,
        )
        if not found:
            raise RuntimeError("Failed to find any available months in archive.")
        start_year, start_month = found

    dest = data_root / "raw" / "klines_1d"
    return download_monthly_1d(
        symbol=symbol,
        market_type=market_type,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        dest_folder=dest,
        progress=progress,
        verbose=verbose,
        print_urls=print_urls,
    )
