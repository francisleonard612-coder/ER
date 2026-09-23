"""
Loads the supplied historical datasets. Schema is inspected at runtime --
nothing about column names/order is assumed. If the files aren't present
(e.g. first deploy before you've uploaded them), the bot logs a warning
and falls back to live-only online learning rather than failing to start.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def load_csv_if_present(path: str, logger) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        logger.warning(f"Historical data file not found, skipping: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to read {path}: {exc!r}")
        return None
    logger.info(f"Loaded {path}: {len(df)} rows, columns={list(df.columns)}")
    return df


def infer_candle_columns(df: pd.DataFrame) -> dict:
    """
    Best-effort column mapping for arbitrary OHLC-ish schemas. Returns a dict
    of canonical_name -> actual_column_name, or None where a column can't be
    confidently identified. Callers must check for None before relying on a
    field -- nothing is fabricated.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def find(*candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    return {
        "symbol": find("symbol", "market", "underlying"),
        "timestamp": find("timestamp", "epoch", "time", "datetime", "date"),
        "open": find("open", "open_price"),
        "high": find("high", "high_price"),
        "low": find("low", "low_price"),
        "close": find("close", "close_price"),
        "volume": find("volume"),
    }
