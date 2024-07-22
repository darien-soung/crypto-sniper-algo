from datetime import datetime, timedelta, timezone
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.models import (
    Exchange,
    Interval,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
)
import numpy as np
import talib
import asyncio
import logging
import colorlog
import cybotrade_indicators
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation
from cybotrade.runtime import StrategyTrader
import pandas as pd
import pandas_ta as ta


class Strategy(BaseStrategy):
    # # Indicator Parameters
    # # macd
    # rsi_crossline_lookback = 5
    # # RSI length
    # rsi_length = 14
    # # SMA length
    # sma_1 = 10
    # sma_2 = 20
    # # Stop Loss candle low length
    # sl_lookback_length = 10
    # # Risk Reward
    # risk_to_reward = 3
    qty = 0.3
    # Time
    time_interval = "15m"

    def __init__(self):
        self.rsi_upper_bound = None
        self.rsi_lower_bound = None
        self.risk_to_reward = None
        self.sl_lookback_length = None
        self.sma_2 = None
        self.sma_1 = None
        self.rsi_length = None
        self.rsi_crossline_lookback = None
        self.sl_max_percentage = None
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("sma1020_macd_rsi_backtest.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "rsi_crossline_lookback":
            self.rsi_crossline_lookback = int(value)
        elif identifier == "rsi_length":
            self.rsi_length = int(value)
        elif identifier == "sma_1":
            self.sma_1 = int(value)
        elif identifier == "sma_2":
            self.sma_2 = int(value)
        elif identifier == "sl_lookback_length":
            self.sl_lookback_length = int(value)
        elif identifier == "risk_to_reward":
            self.risk_to_reward = int(value)
        elif identifier == "rsi_lower_bound":
            self.rsi_lower_bound = int(value)
        elif identifier == "rsi_upper_bound":
            self.rsi_upper_bound = int(value)
        elif identifier == "sl_max_percentage":
            self.sl_max_percentage = float(value)

    async def on_candle_closed(self, strategy, topic, symbol):
        candles = self.data_map[topic]
        start_time = np.array(list(map(lambda c: float(c["start_time"]), candles)))
        open = np.array(list(map(lambda c: float(c["open"]), candles)))
        high = np.array(list(map(lambda c: float(c["high"]), candles)))
        low = np.array(list(map(lambda c: float(c["low"]), candles)))
        close = np.array(list(map(lambda c: float(c["close"]), candles)))
        volume = np.array(list(map(lambda c: float(c["volume"]), candles)))

        # logging.debug(
        #     f"open : {open[-1]}, close: {close[-1]}, high: {high[-1]}, low: {low[-1]}, at {start_time[-1]}"
        # )

        if (
                len(close) < self.sma_1 * 3
                or len(close) < self.sma_2 * 3
                or len(close) < self.sl_lookback_length * 3
                or len(close) < self.rsi_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # Build SMA
        sma_1 = talib.SMA(close, self.sma_1)
        sma_2 = talib.SMA(close, self.sma_2)
        # pandas_sma = ta.sma(close=close, length=self.sma_1)

        # RSI
        rsi = talib.RSI(close, self.rsi_length)
        rsi_crossover = False

        # Value processing

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol, exchange=Exchange.BybitLinear)
        # print("Balance: " + str(wallet_balance))

        ### Entry Conditions
        # Make sure there's no position open
        if position.long.quantity <= 0.0 and position.short.quantity <= 0.0:
            # If SMA 10 flips SMA 20 to the upside
            if sma_1[-1] >= sma_2[-1] and sma_1[-2] <= sma_2[-2]: # and sma_1[-1] * 1 <= close[-1]: # to make sure price is above the MA
                for i in range(np.negative(self.rsi_crossline_lookback + 1), 0):
                    if rsi[i] <= self.rsi_lower_bound:
                        # Enter long position if all conditions fulfilled
                        or_stop_loss = stop_loss = min(low[len(low) - self.sl_lookback_length + 1: len(low) + 1])
                        if (close[-1] - stop_loss) / close[-1] > self.sl_max_percentage:
                            stop_loss = close[-1] * (1 - self.sl_max_percentage)

                        take_profit = close[-1] + (self.risk_to_reward * (close[-1] - or_stop_loss))
                        await strategy.open(side=OrderSide.Buy, quantity=self.qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=or_stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
                        logging.info(
                            f"Placed a buy order with qty {self.qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
                        )
                        break
            # If SMA 10 flips SMA 20 to the downside
            elif sma_1[-1] <= sma_2[-1] and sma_1[-2] >= sma_2[-2]: # and sma_1[-1] * 1 >= close[-1]: # to make sure price is below the MA
                for i in range(np.negative(self.rsi_crossline_lookback + 1), 0):
                    if rsi[i] >= self.rsi_upper_bound:
                        # Enter short position if all conditions fulfilled
                        or_stop_loss = stop_loss = max(high[len(high) - self.sl_lookback_length + 1: len(high) + 1])
                        if (stop_loss - close[-1]) / close[-1] > self.sl_max_percentage:
                            stop_loss = close[-1] * (1 + self.sl_max_percentage)
                        take_profit = close[-1] - (self.risk_to_reward * (or_stop_loss - close[-1]))
                        await strategy.open(side=OrderSide.Sell, quantity=self.qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=or_stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
                        logging.info(
                            f"Placed a sell order with qty {self.qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
                        )
                        break

        ### Exit Conditions
        


config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    active_order_interval=1,
    initial_capital=10000.0,
    candle_topics=["candles-" + Strategy.time_interval + "-BTC/USDT-bybit"],
    start_time=datetime(2023, 6, 9, 0,0,0, tzinfo=timezone.utc),
    end_time=datetime(2024, 6, 10, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=500,
    # exchange_keys="./z_exchange-keys.json",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["rsi_crossline_lookback"] = [5]
hyper_parameters["rsi_lower_bound"] = [30]
hyper_parameters["rsi_upper_bound"] = [70]
hyper_parameters["rsi_length"] = [14]
hyper_parameters["sma_1"] = [10, 20]
hyper_parameters["sma_2"] = [30, 50]
hyper_parameters["sl_lookback_length"] = [10]
hyper_parameters["risk_to_reward"] = [2]
hyper_parameters["sl_max_percentage"] = [0.05]

async def start_backtest():
    await permutation.run(hyper_parameters, Strategy)

asyncio.run(start_backtest())