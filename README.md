# TechnicalResearch

TechnicalResearch is a small Python research project for exploring statistical relationships between technical indicators and daily stock returns in the Vietnamese equity market.

The repository currently focuses on reusable MongoDB data access code and exploratory notebooks. Indicator and return-analysis logic is still early-stage and should be added under `src/` rather than duplicated in notebooks.

## Current Structure

```text
TechnicalResearch/
|-- pyproject.toml
|-- .env.example
|-- src/
|   |-- config.py
|   |-- loaders/
|   |   |-- data_loader.py
|   |   `-- market_data_api.py
|   `-- indicators/
|       |-- momentum.py
|       |-- trend.py
|       |-- volatility.py
|       `-- volume.py
`-- scripts/
    |-- high_liquid.ipynb
    |-- high_liquid.csv
    |-- test_loaders.ipynb
    `-- test_market_data_api_fpt.ipynb
```

Notes:

- `src/loaders/` contains read-only market data loading utilities. The active workflow loads daily bars directly from MongoDB.
- `src/indicators/` exists as the intended home for reusable indicator logic, but the current indicator modules are placeholders.
- `scripts/` contains exploratory notebooks and a small CSV output from the high-liquidity workflow.
- There is no test suite, CLI, report generator, or formal pipeline yet.

## Environment Setup

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
python -m pip install -e .
```

Dependencies are managed in `pyproject.toml`. Do not add a separate `requirements.txt` unless the project intentionally changes dependency management.

Copy `.env.example` to `.env` and adjust local values:

```text
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=vn_market_data
MONGODB_COLLECTION=market_bars_raw
```

## Market Data Sources

The current workflow reads daily OHLCV bars directly from MongoDB through `loaders.data_loader`. The loader is read-only and normalizes nested MongoDB documents into flat rows for pandas or polars.

Expected MongoDB document fields:

```text
_id
meta.symbol
time
open
high
low
close
volume
```

Example:

```python
from loaders.data_loader import load_market_bars, list_symbols

symbols = list_symbols()
df = load_market_bars(
    symbols=["VNM", "FPT"],
    start="2024-01-01",
    end="2025-01-01",
    frame="pandas",
)
```

Returned columns:

```text
_id, time, symbol, open, high, low, close, volume
```

Behavior:

- `start` is inclusive.
- `end` is exclusive.
- `time` is normalized to `YYYY-MM-DD` strings.
- Results are sorted by `meta.symbol`, then `time`.
- The query uses `meta.symbol` and `time` filters so existing MongoDB indexes can be used.
- `frame` may be `"pandas"` or `"polars"`.

MongoDB access in this project must remain read-only.

`loaders.market_data_api` still exists in the repository, but it is no longer the primary documented data path. New research code should call MongoDB directly unless the project intentionally reintroduces the API layer.

## Research Conventions

Sort panel data before any time-series operation:

```python
df = df.sort_values(["symbol", "time"])
```

Forward returns should be computed within each symbol independently:

```python
df["future_return_5"] = (
    df.groupby("symbol")["close"].shift(-5) / df["close"] - 1
)
```

Use features at time `t` only to explain or predict forward returns from `t` to `t+h`. Do not use centered rolling windows, future-filled values, future-shifted indicators, or cross-symbol operations that leak information.

## Notebooks

Current notebooks live in `scripts/`:

- `test_loaders.ipynb`: exploratory checks for the direct MongoDB loader.
- `test_market_data_api_fpt.ipynb`: older exploratory checks for the API loader using FPT data.
- `high_liquid.ipynb`: liquidity-focused exploratory workflow.

Reusable logic discovered in notebooks should be moved into `src/`. Notebooks should stay focused on exploration.

## Current Status

Implemented:

- Environment-based MongoDB configuration in `src/config.py`.
- Direct MongoDB OHLCV loader with pandas/polars output.
- Exploratory notebooks for loader checks and liquidity analysis.

Not implemented yet:

- Reusable technical indicator functions.
- Reusable forward-return construction helpers.
- Feature/label alignment utilities.
- Formal statistical analysis pipeline.
- Automated tests.
