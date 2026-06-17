# TechnicalResearch

TechnicalResearch is a Python research project for studying statistical relationships between common technical indicators and daily stock returns in the Vietnam equity market.

## Objective

The project aims to test whether indicators such as moving averages, momentum, RSI, MACD, volatility, volume-derived measures, and price trend features have measurable relationships with next-day or same-day daily returns for Vietnamese listed equities.

The intended workflow is:

1. Collect and clean Vietnam stock market price and volume data.
2. Compute technical indicators from historical OHLCV data.
3. Align indicators with daily forward returns while avoiding look-ahead bias.
4. Evaluate statistical relationships using correlation, regression, rank tests, factor grouping, and out-of-sample validation.
5. Summarize results in reproducible tables, charts, and research notes.

## Scope

Initial research should focus on liquid Vietnamese stocks listed on HOSE, HNX, and UPCoM, subject to data availability and quality. The analysis should account for common market-data issues such as missing sessions, corporate actions, survivorship bias, thin trading, and extreme outliers.

## Suggested Project Structure

```text
TechnicalResearch/
|-- data/
|   |-- raw/          # Original market data, not committed
|   |-- interim/      # Cleaned intermediate datasets, not committed
|   `-- processed/    # Analysis-ready datasets, not committed
|-- notebooks/        # Exploratory research notebooks
|-- src/              # Reusable Python modules
|-- tests/            # Unit and regression tests
|-- outputs/          # Generated model outputs, not committed
|-- reports/          # Generated research reports, not committed
`-- README.md
```

## Environment Setup

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The initial dependency stack includes:

- `pymongo` for connecting to the existing local MongoDB stock database.
- `pandas` and `polars` for tabular data preparation and analysis.
- `scipy` and `statsmodels` for statistical testing and regression diagnostics.
- `scikit-learn` for modelling and validation workflows.
- `matplotlib`, `seaborn`, and `plotly` for visualization.
- `python-dotenv` for local environment configuration.

Copy `.env.example` to `.env` and adjust the MongoDB database name if needed:

```text
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=technical_research
```

## Methodological Notes

- Use adjusted prices where possible to account for dividends, splits, and other corporate actions.
- Lag technical indicators before comparing them with future returns.
- Separate exploratory analysis from validation to reduce overfitting.
- Report transaction-cost sensitivity before treating any relationship as economically meaningful.
- Compare results across market regimes, liquidity buckets, and listing exchanges.

## Status

This repository is currently in the project setup stage.
