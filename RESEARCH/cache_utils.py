"""
====================================================================================
CACHE UTILITIES MODULE FOR RESEARCH PIPELINE
====================================================================================

This module provides caching functionality for all intermediate data in the
research pipeline. Caching is essential for:

1. SPEED: Avoid recomputing expensive operations (astro calculations, features)
2. REPRODUCIBILITY: Same parameters always produce same cached data
3. GRID SEARCH: Load precomputed features quickly for parameter optimization

CACHE STRUCTURE (FLAT WITH PREFIXES):
─────────────────────────────────────────────────────────────────────────────────
All files are stored in one folder: RESEARCH/cache/
Each file has a descriptive name with prefix:

    {category}__{name}__{hash}.{extension}

Examples:
    market__data__btc_2017-11-01__a3f7b2c1.parquet
    astro__bodies__geo_2017-2024__b4e8c9d2.parquet
    astro__aspects__orb1.0_geo__c5f9d0e3.parquet
    astro__phases__2017-2024__d6a0e1f4.parquet
    features__ternary__orb1.0_geo_phases__e7b1f2a5.parquet
    labels__ternary__h1_gw201__f8c2a3b6.parquet

This structure makes it easy to:
    - See at a glance what each file contains
    - Filter by category prefix (e.g., ls astro__*)
    - Track parameter variations via hash
─────────────────────────────────────────────────────────────────────────────────

USAGE EXAMPLE:
    from RESEARCH.cache_utils import save_cache, load_cache, cache_exists
    
    # Check if cached data exists
    if cache_exists("features", "ternary", params={"orb": 1.0, "coord": "geo"}):
        df = load_cache("features", "ternary", params={"orb": 1.0, "coord": "geo"})
    else:
        df = compute_features(...)
        save_cache(df, "features", "ternary", params={"orb": 1.0, "coord": "geo"})

====================================================================================
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Cache root directory - relative to RESEARCH folder
# We use pathlib to handle cross-platform paths correctly
_CACHE_ROOT: Optional[Path] = None


def get_cache_root() -> Path:
    """
    Get the cache root directory, creating it if necessary.
    
    The cache is stored in RESEARCH/cache/ by default.
    We detect project root by looking for RESEARCH folder in parents.
    
    Returns:
        Path to cache root directory
    """
    global _CACHE_ROOT
    
    if _CACHE_ROOT is not None:
        return _CACHE_ROOT
    
    # Find project root by looking for RESEARCH folder
    current = Path(__file__).resolve()
    
    # Go up the directory tree to find RESEARCH
    for parent in [current] + list(current.parents):
        if parent.name == "RESEARCH":
            _CACHE_ROOT = parent / "cache"
            break
    
    if _CACHE_ROOT is None:
        # Fallback: use current directory
        _CACHE_ROOT = Path.cwd() / "RESEARCH" / "cache"
    
    # Create cache directory if it doesn't exist
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    
    return _CACHE_ROOT


# ==================================================================================
# HASH GENERATION AND HUMAN-READABLE PARAMS
# ==================================================================================

def _params_to_hash(params: Optional[Dict[str, Any]]) -> str:
    """
    Convert parameters dictionary to a unique hash string.
    
    This is used to generate unique filenames for different parameter combinations.
    The hash is deterministic - same params always produce same hash.
    
    Args:
        params: Dictionary of parameters (can contain strings, numbers, lists)
    
    Returns:
        8-character hex hash string
    
    Example:
        params = {"orb_mult": 1.0, "coord_mode": "geo", "bodies": ["Sun", "Moon"]}
        hash = _params_to_hash(params)  # e.g., "a3f7b2c1"
    """
    if params is None or len(params) == 0:
        return "default"
    
    # Sort keys for consistent ordering (important for deterministic hash)
    sorted_params = json.dumps(params, sort_keys=True, default=str)
    
    # Use MD5 for speed (security not important here)
    hash_obj = hashlib.md5(sorted_params.encode("utf-8"))
    
    # Return first 8 characters of hex digest
    return hash_obj.hexdigest()[:8]


def _params_to_readable_suffix(params: Optional[Dict[str, Any]]) -> str:
    """
    Convert key parameters to a short human-readable suffix.
    
    This is added to the filename to make it easier to identify files at a glance.
    Only key parameters are included to keep filename short.
    
    Args:
        params: Dictionary of parameters
    
    Returns:
        Short readable string like "orb1.0_geo" or "btc_2017"
    
    Example:
        params = {"orb_mult": 1.0, "coord_mode": "geo"}
        suffix = _params_to_readable_suffix(params)  # "orb1.0_geo"
    """
    if params is None or len(params) == 0:
        return ""
    
    # Extract key readable values (keep it short!)
    parts = []
    
    # Priority order for different param types
    key_params = [
        ("subject_id", lambda v: str(v)[:10]),
        ("start_date", lambda v: str(v)[:10]),
        ("date_range", lambda v: str(v)[:15]),
        ("coord_mode", lambda v: str(v)),
        ("orb_mult", lambda v: f"orb{v}"),
        ("horizon", lambda v: f"h{v}"),
        ("gauss_window", lambda v: f"gw{v}"),
        ("type", lambda v: str(v)),
        ("include_phases", lambda v: "phases" if v else ""),
    ]
    
    for key, formatter in key_params:
        if key in params:
            val = formatter(params[key])
            if val:  # Skip empty values
                parts.append(val)
    
    # Limit total length
    suffix = "_".join(parts[:4])  # Max 4 parts
    
    # Clean up (remove special chars, limit length)
    suffix = suffix.replace("/", "-").replace(" ", "")[:40]
    
    return suffix


def get_cache_path(
    category: str, 
    name: str, 
    params: Optional[Dict[str, Any]] = None,
    extension: str = "parquet"
) -> Path:
    """
    Generate the full cache file path for given category, name, and parameters.
    
    FILES ARE STORED IN A FLAT STRUCTURE with descriptive prefixes:
        {category}__{name}__{readable_suffix}__{hash}.{extension}
    
    Args:
        category: Cache category prefix, e.g., "market", "astro", "features", "labels"
        name: Data type name, e.g., "data", "bodies", "aspects", "ternary"
        params: Dictionary of parameters that affect the cached data
        extension: File extension, default "parquet" (also supports "pkl", "json")
    
    Returns:
        Full path to cache file
    
    Example:
        path = get_cache_path(
            category="features",
            name="ternary",
            params={"orb_mult": 1.0, "coord_mode": "geo", "include_phases": True}
        )
        # Returns: RESEARCH/cache/features__ternary__orb1.0_geo_phases__a3f7b2c1.parquet
    """
    cache_root = get_cache_root()
    
    # Build filename with clear prefixes
    param_hash = _params_to_hash(params)
    readable_suffix = _params_to_readable_suffix(params)
    
    # Construct filename: category__name__readable__hash.ext
    if readable_suffix:
        filename = f"{category}__{name}__{readable_suffix}__{param_hash}.{extension}"
    else:
        filename = f"{category}__{name}__{param_hash}.{extension}"
    
    return cache_root / filename


# ==================================================================================
# CACHE EXISTENCE CHECK
# ==================================================================================

def cache_exists(
    category: str, 
    name: str, 
    params: Optional[Dict[str, Any]] = None,
    extension: str = "parquet"
) -> bool:
    """
    Check if a cache file exists for the given parameters.
    
    Args:
        category: Cache category (subfolder)
        name: Base name for the cache file
        params: Dictionary of parameters
        extension: File extension
    
    Returns:
        True if cache file exists, False otherwise
    """
    path = get_cache_path(category, name, params, extension)
    return path.exists()


# ==================================================================================
# SAVE FUNCTIONS
# ==================================================================================

def save_cache(
    data: Union[pd.DataFrame, Dict[str, Any]],
    category: str,
    name: str,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Path:
    """
    Save data to cache.
    
    Automatically selects format based on data type:
    - DataFrame -> .parquet (fast binary format, preserves dtypes)
    - Dict/other -> .pkl (pickle, for complex objects)
    
    Args:
        data: Data to cache (DataFrame or dict)
        category: Cache category (subfolder)
        name: Base name for the cache file
        params: Dictionary of parameters that produced this data
        verbose: Print cache path if True
    
    Returns:
        Path to saved cache file
    
    Example:
        # Save feature matrix
        save_cache(
            df_features,
            category="features",
            name="full_features",
            params={"orb_mult": 1.0, "coord_mode": "geo"}
        )
    """
    import pickle
    
    # Determine extension based on data type
    if isinstance(data, pd.DataFrame):
        extension = "parquet"
        path = get_cache_path(category, name, params, extension)
        data.to_parquet(path, index=False)
    else:
        extension = "pkl"
        path = get_cache_path(category, name, params, extension)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    if verbose:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✅ Cached: {path.name} ({size_mb:.2f} MB)")
    
    # Also save params as JSON for human readability
    if params is not None:
        params_path = path.with_suffix(".params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2, default=str)
    
    return path


# ==================================================================================
# LOAD FUNCTIONS
# ==================================================================================

def load_cache(
    category: str,
    name: str,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
    """
    Load data from cache if it exists.
    
    Automatically detects format (.parquet or .pkl).
    
    Args:
        category: Cache category (subfolder)
        name: Base name for the cache file
        params: Dictionary of parameters
        verbose: Print cache path if True
    
    Returns:
        Cached data (DataFrame or dict) or None if not found
    
    Example:
        df = load_cache(
            category="features",
            name="full_features",
            params={"orb_mult": 1.0, "coord_mode": "geo"}
        )
        if df is not None:
            print("Loaded from cache!")
    """
    import pickle
    
    # Try parquet first (most common for DataFrames)
    parquet_path = get_cache_path(category, name, params, "parquet")
    if parquet_path.exists():
        if verbose:
            print(f"📂 Loading from cache: {parquet_path.name}")
        return pd.read_parquet(parquet_path)
    
    # Try pickle (for dicts and complex objects)
    pkl_path = get_cache_path(category, name, params, "pkl")
    if pkl_path.exists():
        if verbose:
            print(f"📂 Loading from cache: {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    
    # No cache found
    return None


# ==================================================================================
# CACHE INVALIDATION
# ==================================================================================

def clear_cache(category: Optional[str] = None, verbose: bool = True) -> int:
    """
    Clear cache files.
    
    In the flat structure, category is used as prefix filter.
    For example, clear_cache("astro") removes all files starting with "astro__".
    
    Args:
        category: If specified, only clear files with this prefix. Otherwise clear all.
        verbose: Print number of files removed
    
    Returns:
        Number of files removed
    """
    cache_root = get_cache_root()
    count = 0
    
    for f in cache_root.iterdir():
        if not f.is_file():
            continue
        
        # Skip non-cache files
        if not (f.suffix in [".parquet", ".pkl", ".json"]):
            continue
        
        # Filter by category prefix if specified
        if category is not None:
            if not f.name.startswith(f"{category}__"):
                continue
        
        f.unlink()
        count += 1
    
    if verbose:
        if category:
            print(f"🗑️  Cleared {count} cache files (category: {category})")
        else:
            print(f"🗑️  Cleared {count} cache files")
    
    return count


def list_cache(category: Optional[str] = None) -> pd.DataFrame:
    """
    List all cached files with their sizes and modification times.
    
    In the flat structure, parses category and name from filename prefix.
    
    Args:
        category: If specified, only list files with this prefix. Otherwise list all.
    
    Returns:
        DataFrame with columns: category, name, params, size_mb, modified
    """
    import datetime
    
    cache_root = get_cache_root()
    rows = []
    
    for f in cache_root.iterdir():
        if not f.is_file():
            continue
        
        # Skip params files and non-cache files
        if f.name.endswith(".params.json") or f.suffix not in [".parquet", ".pkl"]:
            continue
        
        # Parse filename: category__name__readable__hash.ext
        # Example: features__ternary__orb1.0_geo__a3f7b2c1.parquet
        base = f.stem  # Remove extension
        parts = base.split("__")
        
        if len(parts) >= 2:
            file_category = parts[0]
            file_name = parts[1]
            params_readable = parts[2] if len(parts) >= 3 else ""
        else:
            file_category = "unknown"
            file_name = base
            params_readable = ""
        
        # Filter by category if specified
        if category is not None and file_category != category:
            continue
        
        stat = f.stat()
        rows.append({
            "category": file_category,
            "name": file_name,
            "params": params_readable,
            "filename": f.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime)
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["category", "name", "modified"], ascending=[True, True, False])
    
    return df


# ==================================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC DATA TYPES
# ==================================================================================

def cache_market_data(df_market: pd.DataFrame, subject_id: str, start_date: str) -> Path:
    """Cache market data with subject and date range info."""
    params = {"subject_id": subject_id, "start_date": start_date}
    return save_cache(df_market, "market", "market_data", params)


def load_market_data_cached(subject_id: str, start_date: str) -> Optional[pd.DataFrame]:
    """Load cached market data."""
    params = {"subject_id": subject_id, "start_date": start_date}
    return load_cache("market", "market_data", params)


def cache_astro_bodies(df_bodies: pd.DataFrame, coord_mode: str, date_range: str) -> Path:
    """Cache astro body positions."""
    params = {"coord_mode": coord_mode, "date_range": date_range}
    return save_cache(df_bodies, "astro_bodies", "bodies", params)


def load_astro_bodies_cached(coord_mode: str, date_range: str) -> Optional[pd.DataFrame]:
    """Load cached astro body positions."""
    params = {"coord_mode": coord_mode, "date_range": date_range}
    return load_cache("astro_bodies", "bodies", params)


def cache_astro_aspects(df_aspects: pd.DataFrame, orb_mult: float, coord_mode: str, date_range: str) -> Path:
    """Cache astro aspects."""
    params = {"orb_mult": orb_mult, "coord_mode": coord_mode, "date_range": date_range}
    return save_cache(df_aspects, "astro_aspects", "aspects", params)


def load_astro_aspects_cached(orb_mult: float, coord_mode: str, date_range: str) -> Optional[pd.DataFrame]:
    """Load cached astro aspects."""
    params = {"orb_mult": orb_mult, "coord_mode": coord_mode, "date_range": date_range}
    return load_cache("astro_aspects", "aspects", params)


def cache_astro_phases(df_phases: pd.DataFrame, date_range: str) -> Path:
    """Cache moon phases and planet elongations."""
    params = {"date_range": date_range}
    return save_cache(df_phases, "astro_phases", "phases", params)


def load_astro_phases_cached(date_range: str) -> Optional[pd.DataFrame]:
    """Load cached moon phases and planet elongations."""
    params = {"date_range": date_range}
    return load_cache("astro_phases", "phases", params)


def cache_features(df_features: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Cache full feature matrix with config."""
    return save_cache(df_features, "features", "full_features", config)


def load_features_cached(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load cached features."""
    return load_cache("features", "full_features", config)


def cache_labels(df_labels: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Cache labels with labeling config."""
    return save_cache(df_labels, "labels", "labels", config)


def load_labels_cached(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load cached labels."""
    return load_cache("labels", "labels", config)
