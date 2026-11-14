"""
data.py
Shared data stage for the Unit 2 historical-estimation pipelines:
    load (cached CSV, else Yahoo Finance) -> clean -> invariants (log returns).

The first run downloads with yfinance and caches to <unit>/data/<ticker>.csv,
so later runs (and runs without internet) use the cached file.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = UNIT_ROOT / "data"
OUTPUT_DIR = UNIT_ROOT / "outputs"

log = logging.getLogger(__name__)


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker.replace('^', '').replace('/', '_')}.csv"


def load_prices(ticker: str = "^GSPC", start: str = "2005-01-01",
                refresh: bool = False) -> pd.DataFrame:
    """Daily close and volume, indexed by date. Uses the cache unless refresh=True."""
    path = _cache_path(ticker)
    if path.exists() and not refresh:
        log.info("Loading cached prices from %s", path)
        return pd.read_csv(path, index_col="date", parse_dates=True)

    import yfinance as yf  # pip install yfinance
    log.info("Downloading %s from Yahoo Finance (start %s)", ticker, start)
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}: check the ticker or your connection")
    if isinstance(raw.columns, pd.MultiIndex):        # yfinance returns (field, ticker) columns
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close", "Volume"]].rename(columns=str.lower)
    df.index.name = "date"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def clean_prices(df: pd.DataFrame, jump_warn: float = 0.20) -> pd.DataFrame:
    """Basic hygiene before any estimation.
    - sort by date, drop duplicate dates (keep the last print)
    - drop missing / non-positive closes (log returns need positive prices)
    - zero volume -> NaN (usually a data gap, not real zero trading)
    - WARN (don't drop) on daily moves above jump_warn: real crashes exist,
      so a human should check them rather than silently delete them."""
    n0 = len(df)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna() & (df["close"] > 0)].copy()
    if "volume" in df:
        df["volume"] = df["volume"].where(df["volume"] > 0)
    removed = n0 - len(df)
    if removed:
        log.warning("clean_prices: removed %d bad rows", removed)

    r = np.log(df["close"]).diff().abs()
    for date, move in r[r > jump_warn].items():
        log.warning("clean_prices: large move %.1f%% on %s - check it is real",
                    100 * move, date.date())
    return df


def log_returns(close: pd.Series) -> pd.Series:
    """Compounded returns eps_t = ln(v_t / v_{t-1}): the invariants we estimate on."""
    return np.log(close).diff().dropna().rename("ret")
