"""
Postgres connection utilities.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator


@contextmanager
def psql_connection(db_url: str) -> Generator:
    """
    psycopg2 connection context manager.
    """
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError("psycopg2 is not installed. Install it and retry.") from e

    conn = psycopg2.connect(db_url)
    try:
        yield conn
    finally:
        conn.close()
