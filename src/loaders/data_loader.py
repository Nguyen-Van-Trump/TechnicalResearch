from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
import polars as pl
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from config import MongoSettings, load_mongo_settings

FrameType = Literal["pandas", "polars"]

BAR_COLUMNS = ["_id", "time", "symbol", "open", "high", "low", "close", "volume"]


def get_collection(settings: MongoSettings | None = None) -> Collection:
    """Return the configured MongoDB collection."""
    settings = settings or load_mongo_settings()
    client: MongoClient = MongoClient(settings.uri)
    return client[settings.database][settings.collection]


def load_market_bars(
    symbols: str | Iterable[str] | None = None,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    limit: int | None = None,
    frame: FrameType = "pandas",
    collection: Collection | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Load daily OHLCV bars from MongoDB.

    The query is written to use the existing indexes on `meta.symbol` and `time`.
    `start` is inclusive and `end` is exclusive. Date inputs should use
    `YYYY-MM-DD`; returned rows normalize `time` to the same format.
    """
    if collection is None:
        collection = get_collection()
    query = _build_query(symbols=symbols, start=start, end=end)
    projection = {
        "_id": 1,
        "time": 1,
        "meta.symbol": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
    }

    cursor = collection.find(query, projection).sort(
        [("meta.symbol", ASCENDING), ("time", ASCENDING)]
    )
    if limit is not None:
        cursor = cursor.limit(limit)

    rows = [_flatten_bar(document) for document in cursor]

    if frame == "pandas":
        return _to_pandas(rows)
    if frame == "polars":
        return _to_polars(rows)

    raise ValueError("frame must be either 'pandas' or 'polars'")


def list_symbols(collection: Collection | None = None) -> list[str]:
    """Return distinct symbols available in the configured collection."""
    if collection is None:
        collection = get_collection()
    return sorted(symbol for symbol in collection.distinct("meta.symbol") if symbol)


def _build_query(
    *,
    symbols: str | Iterable[str] | None,
    start: date | datetime | str | None,
    end: date | datetime | str | None,
) -> dict[str, Any]:
    query: dict[str, Any] = {}

    if symbols is not None:
        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        query["meta.symbol"] = symbol_list[0] if len(symbol_list) == 1 else {"$in": symbol_list}

    time_filter = {}
    if start is not None:
        time_filter["$gte"] = _coerce_datetime(start)
    if end is not None:
        time_filter["$lt"] = _coerce_datetime(end)
    if time_filter:
        query["time"] = time_filter

    return query


def _flatten_bar(document: dict[str, Any]) -> dict[str, Any]:
    meta = document.get("meta") or {}
    return {
        "_id": str(document.get("_id")),
        "time": _format_date(document.get("time")),
        "symbol": meta.get("symbol"),
        "open": document.get("open"),
        "high": document.get("high"),
        "low": document.get("low"),
        "close": document.get("close"),
        "volume": document.get("volume"),
    }


def _coerce_datetime(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    return datetime.fromisoformat(value)


def _format_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    return str(value)


def _to_pandas(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=BAR_COLUMNS)


def _to_polars(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({column: [] for column in BAR_COLUMNS})
    return pl.DataFrame(rows).select(BAR_COLUMNS)
