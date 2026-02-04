"""
Lightweight artifact cache helpers for pipeline steps.

Each step writes its output plus a sidecar .meta.json with:
- params used
- input file signatures (size + mtime)
- creation time

If params or input signatures change, the cache is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FileSignature:
    """
    Minimal file signature for cache validation.
    """
    path: str
    size: int
    mtime: float


def _signature_for_path(path: Path) -> FileSignature:
    """
    Build a FileSignature from a path.
    """
    stat = path.stat()
    return FileSignature(path=str(path), size=int(stat.st_size), mtime=float(stat.st_mtime))


def _signatures(paths: Iterable[Path]) -> List[FileSignature]:
    """
    Build signatures for existing paths (raises if missing).
    """
    sigs: List[FileSignature] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
        sigs.append(_signature_for_path(p))
    return sigs


def _hash_payload(payload: Dict[str, Any]) -> str:
    """
    Hash a payload dict in a stable way.
    """
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def meta_path_for(output_path: Path) -> Path:
    """
    Sidecar meta path for an output artifact.
    """
    return Path(str(output_path) + ".meta.json")


def build_meta(
    step: str,
    params: Dict[str, Any],
    inputs: Iterable[Path],
    version: int = 1,
) -> Dict[str, Any]:
    """
    Build metadata dict for caching.
    """
    sigs = _signatures(inputs)
    payload = {
        "step": step,
        "params": params,
        "inputs": [s.__dict__ for s in sigs],
        "version": int(version),
    }
    return {
        **payload,
        "key": _hash_payload(payload),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load meta JSON if present.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    """
    Save meta JSON to disk.
    """
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def is_cache_valid(
    output_path: Path,
    params: Dict[str, Any],
    inputs: Iterable[Path],
    step: str,
    version: int = 1,
) -> bool:
    """
    Validate cache based on params + input signatures.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return False

    meta_path = meta_path_for(output_path)
    meta = load_meta(meta_path)
    if not meta:
        return False

    # Basic checks
    if meta.get("step") != step or int(meta.get("version", -1)) != int(version):
        return False

    expected = build_meta(step=step, params=params, inputs=inputs, version=version)
    return meta.get("key") == expected.get("key")
