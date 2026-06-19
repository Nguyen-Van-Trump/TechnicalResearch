from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import polars as pl
from dotenv import load_dotenv


FrameType = Literal["pandas", "polars"]
BAR_COLUMNS = ["time", "symbol", "open", "high", "low", "close", "volume", "source", "dataset"]


def load_market_bars(
    symbols: str | Iterable[str] | None = None,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    limit: int | None = None,
    frame: FrameType = "pandas",
    base_url: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Load daily OHLCV bars from the VietnamMarketDataService API.

    `start` is inclusive and `end` is exclusive. The service owns MongoDB schema
    details; this client only consumes the stable `/api/v1/bars/daily` contract.
    """
    rows = _get_json(
        "/api/v1/bars/daily",
        params={
            "symbols": _format_symbols(symbols),
            "start": _format_date(start),
            "end": _format_date(end),
            "limit": limit,
        },
        base_url=base_url,
        api_key=api_key,
    )["data"]

    if frame == "pandas":
        return pd.DataFrame(rows, columns=BAR_COLUMNS)
    if frame == "polars":
        if not rows:
            return pl.DataFrame({column: [] for column in BAR_COLUMNS})
        return pl.DataFrame(rows).select(BAR_COLUMNS)
    raise ValueError("frame must be either 'pandas' or 'polars'")


def list_symbols(*, base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """Return symbols available through the VietnamMarketDataService API."""
    return _get_json("/api/v1/symbols", base_url=base_url, api_key=api_key)["data"]


def _get_json(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    load_dotenv()
    root = (base_url or os.getenv("VN_MARKET_DATA_API_URL") or "http://127.0.0.1:8000").rstrip("/")
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
    url = f"{root}{path}"
    if query:
        url = f"{url}?{query}"

    headers = {"Accept": "application/json"}
    resolved_api_key = api_key or os.getenv("VN_MARKET_DATA_API_KEY")
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_symbols(symbols: str | Iterable[str] | None) -> str | None:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        return symbols.strip().upper()
    return ",".join(symbol.strip().upper() for symbol in symbols if symbol.strip())


def _format_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
