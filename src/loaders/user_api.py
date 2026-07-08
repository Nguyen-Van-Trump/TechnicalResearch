"""User-facing market data helpers.

Use this module from notebooks or research scripts when you want market data
without dealing with MongoDB collections, projections, or nested document fields.

Examples
--------
Load one symbol:

    from loaders.user_api import load_symbol

    fpt = load_symbol("FPT", start="2024-01-01", end="2025-01-01")

Load multiple symbols:

    from loaders.user_api import load_symbols

    bars = load_symbols(["FPT", "VNM", "HPG"])

Load the saved high-liquidity universe:

    from loaders.user_api import load_high_liquid_symbols, load_symbols

    symbols = load_high_liquid_symbols()
    bars = load_symbols(symbols)

Refresh the saved high-liquidity universe:

    from loaders.user_api import update_high_liquid_symbols

    symbols = update_high_liquid_symbols()

Notes
-----
- Returned bar data is a pandas DataFrame indexed by ``time``.
- The returned bar columns are ``symbol, open, high, low, close, volume``.
- ``start`` is inclusive and ``end`` is exclusive.
- ``update_high_liquid_symbols()`` reads MongoDB but only writes the local CSV
  at ``src/high_liquid.csv`` by default.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from .data_loader import load_market_bars

PathLike: TypeAlias = str | Path

USER_BAR_COLUMNS = ["symbol", "open", "high", "low", "close", "volume"]
DEFAULT_HIGH_LIQUID_PATH = Path(__file__).resolve().parents[1] / "high_liquid.csv"


def load_symbols(
    symbols: str | Iterable[str],
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load OHLCV bars for one or more symbols.

    Parameters
    ----------
    symbols:
        One symbol string, such as ``"FPT"``, or an iterable of symbols.
    start:
        Inclusive start date. Accepts ``YYYY-MM-DD``, ``date``, or ``datetime``.
    end:
        Exclusive end date. Accepts ``YYYY-MM-DD``, ``date``, or ``datetime``.
    limit:
        Optional row limit passed through to the read-only loader.

    Returns
    -------
    pandas.DataFrame
        Data indexed by ``time`` with columns
        ``symbol, open, high, low, close, volume``.
    """
    bars = load_market_bars(
        symbols=symbols,
        start=start,
        end=end,
        limit=limit,
        frame="pandas",
    )
    return _format_user_bars(bars)


def load_symbol(
    symbol: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load OHLCV bars for a single symbol."""
    return load_symbols(symbol, start=start, end=end, limit=limit)


def load_high_liquid_symbols(path: PathLike | None = None) -> list[str]:
    """Read the saved high-liquidity universe and return symbols in file order.

    By default this reads ``src/high_liquid.csv``. Duplicate and blank symbols
    are removed while preserving the first occurrence order.
    """
    universe_path = _resolve_high_liquid_path(path)
    universe = pd.read_csv(universe_path)
    if "symbol" not in universe.columns:
        raise ValueError(f"Missing required 'symbol' column in {universe_path}")

    symbols = universe["symbol"].dropna().astype(str).str.strip()
    return list(dict.fromkeys(symbol for symbol in symbols if symbol))


def update_high_liquid_symbols(
    path: PathLike | None = None,
    *,
    window: int = 10,
    min_avg_volume: float = 1_000_000,
) -> list[str]:
    """Refresh the saved high-liquidity universe and return its symbols.

    The universe rule is the latest available trailing average volume per
    symbol, filtered by ``avg_volume > min_avg_volume``. The default settings
    reproduce the saved CSV contract: 10 trading days and a threshold of
    1,000,000 shares.

    This function does not write to MongoDB. It reads bars from MongoDB through
    the existing read-only loader, then overwrites the local CSV path.
    """
    bars = load_market_bars(frame="pandas")
    high_liquid = _build_high_liquid_universe(
        bars,
        window=window,
        min_avg_volume=min_avg_volume,
    )

    output_path = _resolve_high_liquid_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    high_liquid.to_csv(output_path, index=False)
    return high_liquid["symbol"].tolist()


def _format_user_bars(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    if "_id" in result.columns:
        result = result.drop(columns="_id")

    if "time" not in result.columns:
        raise ValueError("Loaded market bars are missing required 'time' column")

    for column in USER_BAR_COLUMNS:
        if column not in result.columns:
            result[column] = pd.Series(dtype="object")

    result["time"] = pd.to_datetime(result["time"])
    result = result.sort_values(["symbol", "time"]).set_index("time")
    return result.loc[:, USER_BAR_COLUMNS]


def _build_high_liquid_universe(
    bars: pd.DataFrame,
    *,
    window: int,
    min_avg_volume: float,
) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be at least 1")

    volume_data = _prepare_volume_data(bars)
    avg_column = f"avg_volume_{window}d"
    volume_data[avg_column] = (
        volume_data.groupby("symbol", group_keys=False)["volume"]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    latest_volume = (
        volume_data.dropna(subset=[avg_column])
        .sort_values(["symbol", "time"])
        .groupby("symbol", sort=False)
        .tail(1)
        .sort_values(avg_column, ascending=False)
    )
    high_liquid = latest_volume.loc[
        latest_volume[avg_column] > min_avg_volume,
        ["symbol", "time", "volume", avg_column],
    ].copy()

    high_liquid["time"] = high_liquid["time"].dt.date.astype(str)
    return high_liquid.rename(columns={avg_column: "avg_volume_10d"}).reset_index(drop=True)


def _prepare_volume_data(bars: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"symbol", "time", "volume"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Loaded market bars are missing required columns: {missing}")

    volume_data = bars.loc[:, ["symbol", "time", "volume"]].copy()
    volume_data["time"] = pd.to_datetime(volume_data["time"])
    volume_data["volume"] = pd.to_numeric(volume_data["volume"], errors="coerce")
    volume_data = volume_data.dropna(subset=["symbol", "time", "volume"])
    volume_data["symbol"] = volume_data["symbol"].astype(str).str.strip()
    volume_data = volume_data.loc[volume_data["symbol"] != ""]
    return volume_data.sort_values(["symbol", "time"]).reset_index(drop=True)


def _resolve_high_liquid_path(path: PathLike | None) -> Path:
    if path is None:
        return DEFAULT_HIGH_LIQUID_PATH
    return Path(path)


__all__ = [
    "DEFAULT_HIGH_LIQUID_PATH",
    "load_high_liquid_symbols",
    "load_symbol",
    "load_symbols",
    "update_high_liquid_symbols",
]
