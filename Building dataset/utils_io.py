"""
utils_io.py
───────────
Shared I/O helpers used across ETL scripts.
"""

import csv
from pathlib import Path

import pandas as pd
import requests


def ensure_csv(path: Path, columns: list[str]) -> None:
    """Create CSV with header if it doesn't exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_rows_csv(path: Path, rows: list[list], columns: list[str]) -> None:
    """Append rows to a CSV file (create with header if needed)."""
    if not rows:
        return
    path = Path(path)
    ensure_csv(path, columns)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def sanitize_symbol(s) -> str:
    """Uppercase and strip whitespace from a crypto symbol."""
    if s is None:
        return ""
    return str(s).strip().upper()


def http_get(url: str, params: dict = None, timeout: int = 30) -> requests.Response:
    """Simple GET with a reasonable timeout."""
    return requests.get(url, params=params, timeout=timeout)
