from pathlib import Path
import sys

import pandas as pd
import pandas_ta_classic as ta
import mplfinance as mpf
import vectorbt as vbt
from vectorbt.portfolio.enums import StopEntryPrice, StopExitPrice

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loaders.user_api import load_symbols

START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
SYMBOL = "FPT"
EMA_FAST_LENGTH = 20
RISK_REWARD_RATIO = 1.5


def prepare_engulfing_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = (
        load_symbols(symbol, start, end)
        .drop(columns="symbol")
        .reset_index(names="time")
        .sort_values("time", kind="mergesort")
        .reset_index(drop=True)
    )
    df["time"] = pd.to_datetime(df["time"])
    duplicate_bars = int(df["time"].duplicated(keep=False).sum())
    if duplicate_bars:
        print(f"Warning: dropping {duplicate_bars} duplicate-dated bars; keeping last row per date.")
        df = df.drop_duplicates(subset="time", keep="last").reset_index(drop=True)

    ta_accessor = df.ta
    ta_accessor.cores = 0  # pyright: ignore[reportAttributeAccessIssue]
    ta_accessor.cdl_pattern(name="engulfing", append=True)
    ta_accessor.ema(length=EMA_FAST_LENGTH, append=True, col_names=("EMA_fast",))

    return df.set_index("time")


def plot_engulfing_chart(df: pd.DataFrame, symbol: str) -> None:
    bullish_engulfing = df["CDL_ENGULFING"] > 0
    bearish_engulfing = df["CDL_ENGULFING"] < 0

    add_plots = [
        mpf.make_addplot(df["EMA_fast"], color="#1f77b4", width=1.2, label="EMA fast"),
    ]

    if bullish_engulfing.any():
        add_plots.append(
            mpf.make_addplot(
                df["low"].where(bullish_engulfing) * 0.995,
                type="scatter",
                marker="^",
                markersize=90,
                color="#2ca02c",
                label="Bullish engulfing",
            )
        )

    if bearish_engulfing.any():
        add_plots.append(
            mpf.make_addplot(
                df["high"].where(bearish_engulfing) * 1.005,
                type="scatter",
                marker="v",
                markersize=90,
                color="#d62728",
                label="Bearish engulfing",
            )
        )

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        volume=True,
        addplot=add_plots,
        title=f"{symbol} Engulfing Pattern with EMA {EMA_FAST_LENGTH}",
        ylabel="Price",
        ylabel_lower="Volume",
        figratio=(16, 9),
        figscale=1.2,
        warn_too_much_data=2000,
    )


def backtest_engulfing_strategy(
    df: pd.DataFrame,
    *,
    init_cash: float = 100_000_000,
    fees: float = 0.001,
    slippage: float = 0.001,
    risk_reward_ratio: float = RISK_REWARD_RATIO,
) -> vbt.Portfolio:
    stop_distance = (df["close"] - df["low"]) / df["close"]
    valid_stop = stop_distance > 0

    entries = (df["CDL_ENGULFING"] > 0) & (df["close"] > df["EMA_fast"]) & valid_stop
    exits = pd.Series(False, index=df.index)

    sl_stop = stop_distance.where(entries)
    tp_stop = (stop_distance * risk_reward_ratio).where(entries)

    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        exits=exits,
        price=df["close"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        stop_entry_price=StopEntryPrice.Close,
        stop_exit_price=StopExitPrice.StopLimit,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )

    print("\nBacktest parameters:")
    print(f"Entries: {int(entries.sum())}")
    print("Stop loss: entry close to same-day low")
    print(f"Take profit: {risk_reward_ratio:.1f}R")
    print("\nVectorBT Portfolio Performance:")
    print(portfolio.stats())

    return portfolio


def plot_backtest_mpf(
    df: pd.DataFrame,
    portfolio: vbt.Portfolio,
    *,
    risk_reward_ratio: float = RISK_REWARD_RATIO,
) -> None:
    trades = portfolio.trades.records_readable
    closed_trades = trades.loc[trades["Status"] == "Closed"]
    entry_price = pd.Series(index=df.index, dtype="float64")
    exit_price = pd.Series(index=df.index, dtype="float64")

    stop_loss_level = pd.Series(index=df.index, dtype="float64")
    take_profit_level = pd.Series(index=df.index, dtype="float64")
    for _, trade in closed_trades.iterrows():
        entry_time = pd.Timestamp(trade["Entry Timestamp"])
        exit_time = pd.Timestamp(trade["Exit Timestamp"])
        if entry_time not in df.index or exit_time not in df.index:
            continue

        entry_matches = (df.index == entry_time).nonzero()[0]
        exit_matches = (df.index == exit_time).nonzero()[0]
        if len(entry_matches) == 0 or len(exit_matches) == 0:
            continue

        entry_pos = entry_matches[0]
        exit_pos = exit_matches[-1]
        entry_price.iloc[entry_pos] = trade["Avg Entry Price"]

        stop_price = df["low"].iloc[entry_pos]
        entry_close = df["close"].iloc[entry_pos]
        target_price = entry_close + (entry_close - stop_price) * risk_reward_ratio
        stop_loss_level.iloc[entry_pos : exit_pos + 1] = stop_price
        take_profit_level.iloc[entry_pos : exit_pos + 1] = target_price

        if exit_time in exit_price.index:
            exit_price.iloc[exit_pos] = trade["Avg Exit Price"]

    value = portfolio.value()
    pnl = value - value.iloc[0]
    drawdown = portfolio.drawdown() * 100

    add_plots = [
        mpf.make_addplot(df["EMA_fast"], panel=0, color="#1f77b4", width=1.2, label=f"EMA {EMA_FAST_LENGTH}"),
        mpf.make_addplot(
            entry_price,
            panel=0,
            type="scatter",
            marker="^",
            markersize=90,
            color="#2ca02c",
            label="Entry",
        ),
        mpf.make_addplot(
            exit_price,
            panel=0,
            type="scatter",
            marker="v",
            markersize=90,
            color="#9467bd",
            label="Exit",
        ),
        mpf.make_addplot(stop_loss_level, panel=0, color="#d62728", width=1.0, linestyle="--", label="Stop loss"),
        mpf.make_addplot(
            take_profit_level,
            panel=0,
            color="#ff7f0e",
            width=1.0,
            linestyle="--",
            label=f"Take profit {risk_reward_ratio:.1f}R",
        ),
        mpf.make_addplot(pnl, panel=1, color="#2ca02c", width=1.2, ylabel="PnL"),
        mpf.make_addplot(drawdown, panel=2, color="#d62728", width=1.2, ylabel="Drawdown %"),
    ]

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        addplot=add_plots,
        title=f"{SYMBOL} Engulfing Strategy Backtest",
        ylabel="Price",
        panel_ratios=(4, 1.3, 1.3),
        figratio=(16, 10),
        figscale=1.2,
        volume=False,
        warn_too_much_data=2000,
    )


def main() -> None:
    df = prepare_engulfing_data(SYMBOL, START_DATE, END_DATE)
    engulfing_count = int((df["CDL_ENGULFING"] != 0).sum())
    print(f"{SYMBOL}: found {engulfing_count} engulfing candles")
    portfolio = backtest_engulfing_strategy(df)
    plot_backtest_mpf(df, portfolio)
    # plot_engulfing_chart(df, SYMBOL)


if __name__ == "__main__":
    main()
