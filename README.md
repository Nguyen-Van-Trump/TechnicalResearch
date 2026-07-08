# TechnicalResearch

TechnicalResearch is a small Python research project for exploring technical indicators, liquidity filters, and simple backtests on daily Vietnamese equity market data.

The project is intentionally research-oriented. Reusable data loading and calculation helpers live under `src/`; notebooks and one-off experiments live under `notebooks/` or `scripts/`.

## Project Layout

```text
TechnicalResearch/
|-- pyproject.toml
|-- .env.example
|-- src/
|   |-- config.py
|   |-- high_liquid.csv
|   |-- loaders/
|   |   |-- data_loader.py
|   |   |-- market_data_api.py
|   |   `-- user_api.py
|   `-- indicators/
|       |-- momentum.py
|       |-- trend.py
|       |-- volatility.py
|       `-- volume.py
|-- scripts/
|   `-- make_strategies_tutorial.py
|-- notebooks/
|   |-- eda.ipynb
|   |-- high_liquid.ipynb
|   |-- test_loaders.ipynb
|   `-- test_market_data_api_fpt.ipynb
`-- tests/
    `-- test_user_api.py
```

## Setup

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
python -m pip install -e .
```

Dependencies are managed in `pyproject.toml`. Do not add a separate `requirements.txt` unless dependency management is intentionally changed.

`vectorbt[full]` is included and pulls in a large optional dependency stack for backtesting, data integrations, widgets, and distributed execution.

## Configuration

Copy `.env.example` to `.env` and set local values:

```text
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=vn_market_data
MONGODB_COLLECTION=market_bars_raw
```

MongoDB access in this project must remain read-only. Loader code must not insert, update, delete, create indexes, drop collections, or run aggregation stages that write data such as `$out` or `$merge`.

## Data Loading

For notebooks and research scripts, prefer the user-facing loader API:

```python
from loaders.user_api import load_symbol, load_symbols

fpt = load_symbol("FPT", start="2025-01-01", end="2026-01-01")
panel = load_symbols(["FPT", "HPG", "VNM"], start="2025-01-01", end="2026-01-01")
```

Returned bars are pandas DataFrames indexed by `time` with columns:

```text
symbol, open, high, low, close, volume
```

Date behavior:

- `start` is inclusive.
- `end` is exclusive.
- Results are sorted by `symbol, time`.

Lower-level MongoDB access remains available in `loaders.data_loader` for internal use and exploratory checks. It returns `_id` and supports pandas or polars output.

## Liquidity Universe

The saved high-liquidity universe lives at:

```text
src/high_liquid.csv
```

Use:

```python
from loaders.user_api import load_high_liquid_symbols, update_high_liquid_symbols

symbols = load_high_liquid_symbols()
symbols = update_high_liquid_symbols()
```

The default update rule is:

- compute trailing 10-trading-day average volume per symbol,
- keep the latest valid row per symbol,
- filter `avg_volume_10d > 1_000_000`,
- save `symbol,time,volume,avg_volume_10d` back to `src/high_liquid.csv`.

The update reads from MongoDB but only writes the local CSV.

## Strategy Tutorial

Run the current strategy tutorial:

```powershell
.\.venv\Scripts\python.exe scripts\make_strategies_tutorial.py
```

The script:

- loads FPT daily bars,
- applies a `pandas_ta_classic` strategy with trend, momentum, and volatility indicators,
- disables pandas-ta multiprocessing for Windows stability,
- builds basic bullish, bearish, and strong-bullish signals,
- backtests the strong signal,
- prints cumulative return and risk metrics,
- plots cumulative return and drawdown.

If running in a non-interactive environment, use:

```powershell
$env:MPLBACKEND='Agg'; .\.venv\Scripts\python.exe scripts\make_strategies_tutorial.py
```

## Research Conventions

Sort panel data before time-series operations:

```python
df = df.sort_values(["symbol", "time"])
```

Calculate forward returns within each symbol:

```python
df["future_return_5"] = (
    df.groupby("symbol")["close"].shift(-5) / df["close"] - 1
)
```

Avoid lookahead bias:

- features at time `t` may only use information available at or before `t`,
- forward returns may use future prices only as labels,
- never mix symbols in shifts, rolling windows, indicators, or returns,
- avoid centered rolling windows, future-filled features, and future-shifted indicators.

Be careful with duplicate symbol/date rows. The current MongoDB collection may contain multiple datasets for the same symbol and date. For production-quality research, explicitly choose a source/dataset or otherwise deduplicate before indicator and backtest calculations.

## Tests

Run the current unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_user_api
```

The tests cover the user-facing loader API using mocked market data, so they do not require a live MongoDB connection.

## Current Status

Implemented:

- environment-based MongoDB settings in `src/config.py`,
- read-only MongoDB loader in `src/loaders/data_loader.py`,
- user-facing data API in `src/loaders/user_api.py`,
- saved and refreshable high-liquidity universe,
- strategy tutorial script,
- basic tests for `user_api`.

Still evolving:

- reusable indicator modules under `src/indicators/`,
- forward-return construction helpers,
- feature/label alignment utilities,
- formal statistical analysis pipeline.
