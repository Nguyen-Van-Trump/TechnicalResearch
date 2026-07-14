from pathlib import Path
import sys

import pandas_ta_classic as ta
import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loaders.user_api import load_symbols

START_DATE = "2025-01-01"
END_DATE = "2026-01-01"
SYMBOL = "HPG"

# Create custom strategy
my_strategy = ta.Strategy(
 name="TrendMomentumVolatility",
 description="Combines trend, momentum, and volatility indicators",
 ta=[
 # Trend Indicators
 {"kind": "sma", "length": 10},
 {"kind": "sma", "length": 50},
 {"kind": "ema", "length": 12},
 {"kind": "ema", "length": 26},
 
 # Momentum Indicators
 {"kind": "rsi", "length": 14},
 {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
 {"kind": "stoch", "k": 14, "d": 3},
 
 # Volatility Indicators
 {"kind": "bbands", "length": 20, "std": 2},
 {"kind": "atr", "length": 14},
 {"kind": "kc", "length": 20},
 ]
)

def backtest_strategy_vbt(df: pd.DataFrame, entries: pd.Series, exits: pd.Series) -> None:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').set_index('time')

    # Create a vectorbt portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,  # 0.1% trading fees
        slippage=0.001,  # 0.1% slippage
        freq='1D'
    )

    print(f"\nVectorBT Portfolio Performance:")
    print(portfolio.stats())

    # Plot the portfolio performance
    portfolio.plot().show() # pyright: ignore[reportOptionalMemberAccess]

def main() -> None:
    df = (
        load_symbols(SYMBOL, START_DATE, END_DATE)
        .drop("symbol", axis=1)
        .reset_index(names="time")
    )

    print(f"Strategy: {my_strategy.name}")
    print(f"Number of indicators: {len(my_strategy.ta)}")

    # Keep one accessor instance. Re-accessing df.ta creates a fresh accessor
    # with default multiprocessing cores.
    ta_accessor = df.ta
    ta_accessor.cores = 0 # pyright: ignore[reportAttributeAccessIssue]
    ta_accessor.strategy(my_strategy)

    # Check new columns
    new_columns = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
    print(f"\nAdded {len(new_columns)} indicator columns:")
    print(new_columns)

    # Generate entry signals (MA crossover + RSI confirmation)
    entries = (
    (df['SMA_10'] > df['SMA_50']) & # Fast MA above slow MA
    (df['RSI_14'] < 70) # RSI below overbought
    )

    # Generate exit signals
    exits = (
    (df['SMA_10'] < df['SMA_50']) | # Fast MA below slow MA
    (df['RSI_14'] > 70) # RSI extremely overbought
    )

    backtest_strategy_vbt(df, entries, exits)


if __name__ == "__main__":
    main()
