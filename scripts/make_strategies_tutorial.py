from pathlib import Path
import sys

import pandas_ta_classic as ta
from pandas_ta_classic import (
 percent_return, 
 log_return, 
 cagr,
 calmar_ratio,
 sharpe_ratio,
 sortino_ratio
)
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
SYMBOL = "FPT"

# Create custom strategy
my_strategy = ta.Strategy(
 name="TrendMomentumVolatility",
 description="Combines trend, momentum, and volatility indicators",
 ta=[
 # Trend Indicators
 {"kind": "sma", "length": 20},
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

def backtest_strategy(df: pd.DataFrame, signal_column: str = 'strategy_signal') -> None:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').set_index('time')

    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['returns'] * df[signal_column].shift(1)
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()

    print(f"\nCumulative returns for {SYMBOL} from {START_DATE} to {END_DATE}:")
    print(f"Buy and Hold: {df['cumulative_returns'].iloc[-1]:.2%}")
    print(f"Strategy: {df['cumulative_strategy_returns'].iloc[-1]:.2%}")

    # Calculate metrics on strategy equity curve
    strategy_equity = (
        df['close'] * df['cumulative_strategy_returns'] / df['cumulative_returns']
    ).dropna()

    # CAGR (Compound Annual Growth Rate)
    strategy_cagr = ta.cagr(strategy_equity)
    print(f"\nCAGR: {strategy_cagr * 100:.2f}%")

    # Sharpe Ratio
    strategy_sharpe = ta.sharpe_ratio(strategy_equity)
    print(f"Sharpe Ratio: {strategy_sharpe:.2f}")

    # Calmar Ratio
    strategy_calmar = ta.calmar_ratio(strategy_equity)
    print(f"Calmar Ratio: {strategy_calmar:.2f}")

    running_max = df['cumulative_strategy_returns'].expanding().max()
    drawdown = (df['cumulative_strategy_returns'] - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"\nMax Drawdown: {max_drawdown * 100:.2f}%")

    # Calculate win rate
    winning_trades = df[df['strategy_returns'] > 0]
    losing_trades = df[df['strategy_returns'] < 0]
    try:
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100
    except ZeroDivisionError:
        win_rate = 0.0
    avg_win = winning_trades['strategy_returns'].mean() * 100
    avg_loss = losing_trades['strategy_returns'].mean() * 100

    print(f"\n{'='*50}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {abs(avg_win / avg_loss):.2f}")
    print(f"{'='*50}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot cumulative returns
    ax1.plot(df.index, df['cumulative_returns'], label='Buy & Hold', linewidth=2)
    ax1.plot(df.index, df['cumulative_strategy_returns'], label='Strategy', linewidth=2)
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot drawdown
    ax2.fill_between(df.index, 0, drawdown * 100, color='red', alpha=0.3)
    ax2.set_title('Strategy Drawdown')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

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
    ta_accessor.cores = 0
    ta_accessor.strategy(my_strategy)

    # Keep warmup rows for indicators, but analyze only the requested period.
    df = df.loc[df["time"] >= START_DATE].copy()

    # Check new columns
    new_columns = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
    print(f"\nAdded {len(new_columns)} indicator columns:")
    print(new_columns)

    # Define bullish conditions
    bullish_conditions = (
    (df['close'] > df['SMA_20']) & # Price above 20 SMA
    (df['SMA_20'] > df['SMA_50']) & # Short MA above long MA
    (df['RSI_14'] > 30) & (df['RSI_14'] < 70) & # RSI in neutral zone
    (df['close'] > df['BBL_20_2.0']) # Price above lower Bollinger Band
    )

    # Define bearish conditions
    bearish_conditions = (
    (df['close'] < df['SMA_20']) & # Price below 20 SMA
    (df['SMA_20'] < df['SMA_50']) & # Short MA below long MA
    (df['RSI_14'] > 30) & (df['RSI_14'] < 70) & # RSI in neutral zone
    (df['close'] < df['BBU_20_2.0']) # Price below upper Bollinger Band
    )

    # Create signals
    df['strategy_signal'] = 0
    df.loc[bullish_conditions, 'strategy_signal'] = 1
    df.loc[bearish_conditions, 'strategy_signal'] = -1

    print(f"Bullish signals: {(df['strategy_signal'] == 1).sum()}")
    print(f"Bearish signals: {(df['strategy_signal'] == -1).sum()}")

    # Calculate signal strength
    df['signal_strength'] = 0

    # Add points for each bullish condition
    df.loc[df['close'] > df['SMA_20'], 'signal_strength'] += 1
    df.loc[df['SMA_20'] > df['SMA_50'], 'signal_strength'] += 1
    df.loc[df['RSI_14'] > 50, 'signal_strength'] += 1
    df.loc[df['MACDh_12_26_9'] > 0, 'signal_strength'] += 1

    # Filter for strong signals only
    strong_bullish = df[(df['strategy_signal'] == 1) & (df['signal_strength'] >= 3)]
    print(f"Strong bullish signals: {len(strong_bullish)}")

    df['strong_signal'] = 0
    df.loc[strong_bullish.index, 'strong_signal'] = 1

    backtest_strategy(df, signal_column='strong_signal')


if __name__ == "__main__":
    main()
