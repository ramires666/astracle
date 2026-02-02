"""
Astro data step: bodies, aspects, and optional transit aspects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.astro.engine.aspects import calculate_aspects, calculate_transit_aspects
from src.astro.engine.calculator import calculate_bodies, calculate_daily_bodies, set_ephe_path
from src.astro.engine.models import AspectConfig, BodyPosition
from src.astro.engine.settings import AstroSettings
from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext
from src.pipeline.astro_xgb.naming import orb_tag


def _iter_with_progress(items: Iterable, enabled: bool, desc: str):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        print("[WARN] tqdm is not installed, progress disabled.")
        return items
    return tqdm(items, desc=desc, unit="day")


def _parse_birth_dt_utc(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _scale_aspects(aspects: List[AspectConfig], orb_mult: float) -> List[AspectConfig]:
    scaled: List[AspectConfig] = []
    for a in aspects:
        scaled.append(AspectConfig(name=a.name, degree=a.degree, orb=float(a.orb) * float(orb_mult)))
    return scaled


def _bodies_by_date_from_df(df_bodies: pd.DataFrame) -> Dict[pd.Timestamp, List[BodyPosition]]:
    out: Dict[pd.Timestamp, List[BodyPosition]] = {}
    df = df_bodies.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for d, grp in df.groupby("date"):
        bodies: List[BodyPosition] = []
        for _, row in grp.iterrows():
            bodies.append(
                BodyPosition(
                    date=row["date"],
                    body=row["body"],
                    lon=float(row["lon"]),
                    lat=float(row["lat"]),
                    speed=float(row["speed"]),
                    is_retro=bool(row["is_retro"]),
                    sign=row["sign"],
                    declination=float(row["declination"]),
                )
            )
        out[pd.Timestamp(d).date()] = bodies
    return out


def _build_center(
    ctx: PipelineContext,
    center: str,
    market_daily_path: Path,
    settings: AstroSettings,
    time_utc,
    orb: float,
    include_pair_aspects: bool,
    include_transit_aspects: bool,
    force: bool,
    progress: bool,
    suffix: str = "",
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Build astro data for a single center (geo or helio).
    """
    tag = orb_tag(orb)
    bodies_path = ctx.processed_dir / f"{ctx.subject.subject_id}_astro_bodies{suffix}.parquet"
    aspects_path = ctx.processed_dir / f"{ctx.subject.subject_id}_astro_aspects_{tag}{suffix}.parquet"
    transits_path = ctx.processed_dir / f"{ctx.subject.subject_id}_transit_aspects_{tag}{suffix}.parquet"

    df_market = pd.read_parquet(market_daily_path)
    dates = pd.to_datetime(df_market["date"]).dt.date

    bodies_params = {"center": center, "daily_time_utc": str(time_utc)}
    if not force and is_cache_valid(bodies_path, params=bodies_params, inputs=[market_daily_path], step="bodies"):
        df_bodies = pd.read_parquet(bodies_path)
        bodies_by_date = _bodies_by_date_from_df(df_bodies)
        print(f"[ASTRO] Using cached bodies ({center}):", bodies_path)
    else:
        bodies_rows: List[dict] = []
        bodies_by_date: Dict[pd.Timestamp, List[BodyPosition]] = {}
        for d in _iter_with_progress(dates, progress, desc=f"astro bodies {center}"):
            bodies = calculate_daily_bodies(d, time_utc, settings.bodies, center=center)
            bodies_by_date[d] = bodies
            for b in bodies:
                bodies_rows.append(
                    {
                        "date": b.date,
                        "body": b.body,
                        "lon": b.lon,
                        "lat": b.lat,
                        "speed": b.speed,
                        "is_retro": b.is_retro,
                        "sign": b.sign,
                        "declination": b.declination,
                    }
                )
        df_bodies = pd.DataFrame(bodies_rows)
        df_bodies.to_parquet(bodies_path, index=False)
        meta = build_meta(step="bodies", params=bodies_params, inputs=[market_daily_path])
        save_meta(meta_path_for(bodies_path), meta)
        print(f"[ASTRO] Built bodies ({center}):", bodies_path)
        print(f"[ASTRO] Bodies rows: {len(df_bodies)}")

    df_aspects = pd.DataFrame(columns=["date", "p1", "p2", "aspect", "orb", "is_exact", "is_applying"])
    if include_pair_aspects:
        aspect_params = {"orb_multiplier": orb, "center": center}
        if not force and is_cache_valid(aspects_path, params=aspect_params, inputs=[bodies_path], step="aspects"):
            df_aspects = pd.read_parquet(aspects_path)
            print(f"[ASTRO] Using cached aspects ({center}):", aspects_path)
        else:
            aspects_rows: List[dict] = []
            aspects_cfg = _scale_aspects(settings.aspects, orb)
            for d, bodies in _iter_with_progress(bodies_by_date.items(), progress, desc=f"astro aspects {center}"):
                hits = calculate_aspects(bodies, aspects_cfg)
                for a in hits:
                    aspects_rows.append(
                        {
                            "date": a.date,
                            "p1": a.p1,
                            "p2": a.p2,
                            "aspect": a.aspect,
                            "orb": a.orb,
                            "is_exact": a.is_exact,
                            "is_applying": a.is_applying,
                        }
                    )
            df_aspects = pd.DataFrame(aspects_rows)
            df_aspects.to_parquet(aspects_path, index=False)
            meta = build_meta(step="aspects", params=aspect_params, inputs=[bodies_path])
            save_meta(meta_path_for(aspects_path), meta)
            print(f"[ASTRO] Built aspects ({center}):", aspects_path)
            print(f"[ASTRO] Aspects rows: {len(df_aspects)}")

    df_transits = None
    if include_transit_aspects:
        transit_params = {"orb_multiplier": orb, "center": center}
        if not force and is_cache_valid(transits_path, params=transit_params, inputs=[bodies_path], step="transits"):
            df_transits = pd.read_parquet(transits_path)
            print(f"[ASTRO] Using cached transits ({center}):", transits_path)
        else:
            birth_dt = _parse_birth_dt_utc(ctx.subject.birth_dt_utc)
            natal_bodies = calculate_bodies(birth_dt, settings.bodies, center=center)
            rows: List[dict] = []
            aspects_cfg = _scale_aspects(settings.aspects, orb)
            for d, bodies in _iter_with_progress(bodies_by_date.items(), progress, desc=f"astro transits {center}"):
                hits = calculate_transit_aspects(bodies, natal_bodies, aspects_cfg)
                for h in hits:
                    rows.append(
                        {
                            "date": h.date,
                            "transit_body": h.transit_body,
                            "natal_body": h.natal_body,
                            "aspect": h.aspect,
                            "orb": h.orb,
                            "is_exact": h.is_exact,
                            "is_applying": h.is_applying,
                        }
                    )
            df_transits = pd.DataFrame(rows)
            df_transits.to_parquet(transits_path, index=False)
            meta = build_meta(step="transits", params=transit_params, inputs=[bodies_path])
            save_meta(meta_path_for(transits_path), meta)
            print(f"[ASTRO] Built transits ({center}):", transits_path)
            print(f"[ASTRO] Transits rows: {len(df_transits)}")

    if not df_bodies.empty:
        print(f"[ASTRO] Date range ({center}): "
              f"{pd.to_datetime(df_bodies['date']).min().date()} -> "
              f"{pd.to_datetime(df_bodies['date']).max().date()}")
    return bodies_path, aspects_path, transits_path if include_transit_aspects else None


def run_astro_step(
    ctx: PipelineContext,
    market_daily_path: Path,
    force: bool = False,
    orb_multiplier: float = 1.0,
    include_pair_aspects: Optional[bool] = None,
    include_transit_aspects: Optional[bool] = None,
    progress: bool = True,
    include_both_centers: Optional[bool] = None,
) -> Dict[str, Tuple[Path, Path, Optional[Path]]]:
    """
    Build astro bodies/aspects/transits for one or both centers.
    Returns dict: {center: (bodies_path, aspects_path, transits_path)}
    """
    astro_cfg = ctx.cfg_astro.get("astro", {})

    include_pair_aspects = bool(
        astro_cfg.get("include_pair_aspects", True) if include_pair_aspects is None else include_pair_aspects
    )
    include_transit_aspects = bool(
        astro_cfg.get("include_transit_aspects", False)
        if include_transit_aspects is None
        else include_transit_aspects
    )
    include_both_centers = bool(
        astro_cfg.get("include_both_centers", False)
        if include_both_centers is None
        else include_both_centers
    )

    ephe_path = Path(astro_cfg["ephe_path"])
    if not ephe_path.is_absolute():
        ephe_path = ctx.root / ephe_path
    set_ephe_path(str(ephe_path))

    settings = AstroSettings(
        bodies_path=ctx.root / astro_cfg["bodies_path"],
        aspects_path=ctx.root / astro_cfg["aspects_path"],
    )
    time_utc = datetime.strptime(astro_cfg["daily_time_utc"], "%H:%M:%S").time()
    center = astro_cfg.get("center", "geo")

    orb = float(orb_multiplier)
    results: Dict[str, Tuple[Path, Path, Optional[Path]]] = {}

    if include_both_centers:
        for c in ["geo", "helio"]:
            suffix = f"_{c}"
            results[c] = _build_center(
                ctx=ctx,
                center=c,
                market_daily_path=market_daily_path,
                settings=settings,
                time_utc=time_utc,
                orb=orb,
                include_pair_aspects=include_pair_aspects,
                include_transit_aspects=include_transit_aspects,
                force=force,
                progress=progress,
                suffix=suffix,
            )
    else:
        results[center] = _build_center(
            ctx=ctx,
            center=center,
            market_daily_path=market_daily_path,
            settings=settings,
            time_utc=time_utc,
            orb=orb,
            include_pair_aspects=include_pair_aspects,
            include_transit_aspects=include_transit_aspects,
            force=force,
            progress=progress,
            suffix="",
        )

    return results
