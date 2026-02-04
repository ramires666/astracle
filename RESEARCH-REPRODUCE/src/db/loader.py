"""
Load data into Postgres.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def _require_psycopg2():
    """
    Ensure psycopg2 is available.
    """
    try:
        import psycopg2  # noqa: F401
    except ImportError as e:
        raise RuntimeError("psycopg2 is not installed. Install it and retry.") from e


def upsert_subject_psql(subject, db_url: str) -> None:
    """
    Insert or update a record in subjects table.
    """
    _require_psycopg2()
    import psycopg2

    sql = """
    INSERT INTO subjects (subject_id, symbol, exchange, birth_dt_utc, birth_lat, birth_lon)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (subject_id) DO UPDATE
      SET symbol = EXCLUDED.symbol,
          exchange = EXCLUDED.exchange,
          birth_dt_utc = EXCLUDED.birth_dt_utc,
          birth_lat = EXCLUDED.birth_lat,
          birth_lon = EXCLUDED.birth_lon
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    subject.subject_id,
                    subject.symbol,
                    subject.exchange,
                    subject.birth_dt_utc,
                    subject.birth_lat,
                    subject.birth_lon,
                ),
            )


def insert_astro_bodies_psql(
    df: pd.DataFrame,
    subject_id: str,
    db_url: str,
    table: str = "astro_bodies_daily",
    page_size: int = 1000,
) -> None:
    """
    Load daily body positions into Postgres.
    """
    _require_psycopg2()
    import psycopg2
    from psycopg2.extras import execute_values

    rows = [
        (
            subject_id,
            row["date"],
            row["body"],
            float(row["lon"]),
            float(row["lat"]),
            float(row["speed"]),
            bool(row["is_retro"]),
            row["sign"],
            float(row["declination"]),
        )
        for _, row in df.iterrows()
    ]

    sql = f"""
    INSERT INTO {table}
      (subject_id, date, body, lon, lat, speed, is_retro, sign, declination)
    VALUES %s
    ON CONFLICT (subject_id, date, body) DO UPDATE
      SET lon = EXCLUDED.lon,
          lat = EXCLUDED.lat,
          speed = EXCLUDED.speed,
          is_retro = EXCLUDED.is_retro,
          sign = EXCLUDED.sign,
          declination = EXCLUDED.declination
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)


def insert_astro_aspects_psql(
    df: pd.DataFrame,
    subject_id: str,
    db_url: str,
    table: str = "astro_aspects_daily",
    page_size: int = 1000,
) -> None:
    """
    Load daily aspects into Postgres.
    """
    _require_psycopg2()
    import psycopg2
    from psycopg2.extras import execute_values

    rows = [
        (
            subject_id,
            row["date"],
            row["p1"],
            row["p2"],
            row["aspect"],
            float(row["orb"]),
            bool(row["is_exact"]),
            bool(row["is_applying"]),
        )
        for _, row in df.iterrows()
    ]

    sql = f"""
    INSERT INTO {table}
      (subject_id, date, p1, p2, aspect, orb, is_exact, is_applying)
    VALUES %s
    ON CONFLICT (subject_id, date, p1, p2, aspect) DO UPDATE
      SET orb = EXCLUDED.orb,
          is_exact = EXCLUDED.is_exact,
          is_applying = EXCLUDED.is_applying
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)


def insert_natal_bodies_psql(
    df: pd.DataFrame,
    subject_id: str,
    db_url: str,
    table: str = "natal_bodies",
    page_size: int = 1000,
) -> None:
    """
    Load natal bodies into Postgres.
    """
    _require_psycopg2()
    import psycopg2
    from psycopg2.extras import execute_values

    rows = [
        (
            subject_id,
            row["body"],
            float(row["lon"]),
            float(row["lat"]),
            float(row["speed"]),
            bool(row["is_retro"]),
            row["sign"],
            float(row["declination"]),
        )
        for _, row in df.iterrows()
    ]

    sql = f"""
    INSERT INTO {table}
      (subject_id, body, lon, lat, speed, is_retro, sign, declination)
    VALUES %s
    ON CONFLICT (subject_id, body) DO UPDATE
      SET lon = EXCLUDED.lon,
          lat = EXCLUDED.lat,
          speed = EXCLUDED.speed,
          is_retro = EXCLUDED.is_retro,
          sign = EXCLUDED.sign,
          declination = EXCLUDED.declination
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)


def insert_natal_aspects_psql(
    df: pd.DataFrame,
    subject_id: str,
    db_url: str,
    table: str = "natal_aspects",
    page_size: int = 1000,
) -> None:
    """
    Load natal aspects into Postgres.
    """
    _require_psycopg2()
    import psycopg2
    from psycopg2.extras import execute_values

    rows = [
        (
            subject_id,
            row["p1"],
            row["p2"],
            row["aspect"],
            float(row["orb"]),
        )
        for _, row in df.iterrows()
    ]

    sql = f"""
    INSERT INTO {table}
      (subject_id, p1, p2, aspect, orb)
    VALUES %s
    ON CONFLICT (subject_id, p1, p2, aspect) DO UPDATE
      SET orb = EXCLUDED.orb
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)


def insert_transit_aspects_psql(
    df: pd.DataFrame,
    subject_id: str,
    db_url: str,
    table: str = "transit_aspects_daily",
    page_size: int = 1000,
) -> None:
    """
    Load transit aspects (transit -> natal) into Postgres.
    """
    _require_psycopg2()
    import psycopg2
    from psycopg2.extras import execute_values

    rows = [
        (
            subject_id,
            row["date"],
            row["transit_body"],
            row["natal_body"],
            row["aspect"],
            float(row["orb"]),
            bool(row["is_exact"]),
            bool(row["is_applying"]),
        )
        for _, row in df.iterrows()
    ]

    sql = f"""
    INSERT INTO {table}
      (subject_id, date, transit_body, natal_body, aspect, orb, is_exact, is_applying)
    VALUES %s
    ON CONFLICT (subject_id, date, transit_body, natal_body, aspect) DO UPDATE
      SET orb = EXCLUDED.orb,
          is_exact = EXCLUDED.is_exact,
          is_applying = EXCLUDED.is_applying
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=page_size)
