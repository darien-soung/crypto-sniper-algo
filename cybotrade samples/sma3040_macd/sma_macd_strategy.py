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
import pandas as pd


class Strategy(BaseStrategy):
    # Indicator Parameters
    # macd
    macd_slow_line = 26
    macd_fast_line = 12
    macd_signal = 9
    macd_crossline_lookback = 8
    # SMA lengths
    sma_1 = 30
    sma_2 = 40
    # Stop Loss config
    sl_low_length = 10
    sl_max_percentage = 0.05 # example: 0.05 = 5%
    # To keep track of the buy/sell flat
    can_entry_buy = True
    can_entry_sell = True
    # Risk Reward
    risk_to_reward = 3
    qty = 0.75
    btcount = 0
    # Time
    time_interval = "30m"


    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler("log_sma_macd_strategy_backtest.log", when="h")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "macd_slow_line":
            self.macd_slow_line = int(value)
        elif identifier == "macd_fast_line":
            self.macd_fast_line = int(value)
        elif identifier == "macd_signal":
            self.macd_signal = int(value)
        elif identifier == "sma_1":
            self.sma_1 = int(value)
        elif identifier == "sma_2":
            self.sma_2 = int(value)
        elif identifier == "sl_low_length":
            self.sl_low_length = int(value)
        elif identifier == "macd_crossline_lookback":
            self.macd_crossline_lookback = int(value)

    async def on_candle_closed(self, strategy, topic, symbol):
        candles = self.data_map[topic]
        start_time = np.array(list(map(lambda c: float(c["start_time"]), candles)))
        open = np.array(list(map(lambda c: float(c["open"]), candles)))
        high = np.array(list(map(lambda c: float(c["high"]), candles)))
        low = np.array(list(map(lambda c: float(c["low"]), candles)))
        close = np.array(list(map(lambda c: float(c["close"]), candles)))
        volume = np.array(list(map(lambda c: float(c["volume"]), candles)))

        # Adding code into my own DF
        # new_row = {'start_time': int(start_time[-1]), 'low': str(low[-1]), 'close': str(close[-1])}
        # df.append(new_row)
        # print("Start time: " + str((int(start_time[-1]))))
        # print("Close price: " + str(close[-1]))

        logging.debug(
            f"open : {open[-1]}, close: {close[-1]}, high: {high[-1]}, low: {low[-1]}, at {start_time[-1]}"
        )

        if (
                len(close) < self.macd_slow_line * 3
                or len(close) < self.macd_fast_line * 3
                or len(close) < self.macd_signal * 3
                or len(close) < self.sma_1 * 3
                or len(close) < self.sma_2 * 3
                or len(close) < self.sl_low_length * 3
        ):
            logging.info("Not enough candles to calculate indicators")
            return

        # SMA
        sma_30 = talib.SMA(close, self.sma_1)
        sma_40 = talib.SMA(close, self.sma_2)
        # MACD
        macd = talib.MACD(close, self.macd_fast_line, self.macd_slow_line, self.macd_signal)
        macd_line = macd[0]
        signal_line = macd[1]
        macd_crossover = False

        position = await strategy.position(symbol=symbol, exchange=Exchange.BybitLinear)
        # wallet_balance = await strategy.get_current_available_balance(symbol=symbol)
        qty = 0.75 # self.qty  # qty of BTC to enter tclarade


        ### Strategy conditions
        # Make sure there's no position open
        if position.long.quantity <= 0.0 and position.short.quantity <= 0.0:
            # If sma 30 flips sma 40 to the upside
            if sma_30[-1] >= sma_40[-1] and sma_30[-2] < sma_40[-2] and sma_30[-1] * 1 < close[-1]:
                # Check for MACD crossover in the last 10 data points
                for i in range(np.negative(self.macd_crossline_lookback)+1, 0): # -5, -4, -3, -2, -1
                    if macd_line[i] > 0 >= macd_line[i - 1]:
                        macd_crossover = True
                        break
                # Enter position if crossover happened recently
                if macd_crossover is True:
                    or_stop_loss = stop_loss = min(low[len(low) - self.sl_low_length: len(low)]) # Set stop loss on the low for the previous N number of candles
                    if (close[-1] - stop_loss) / close[-1] > self.sl_max_percentage:
                        stop_loss = close[-1] * (1 - self.sl_max_percentage)
                    take_profit = close[-1] + (self.risk_to_reward * (close[-1] - or_stop_loss))
                    await strategy.open(side=OrderSide.Buy, quantity=qty, is_hedge_mode=False, take_profit=take_profit, stop_loss=stop_loss, symbol=symbol, exchange=Exchange.BybitLinear, is_post_only=False)
                    logging.info(
                        f"Placed a buy order with qty {qty} when close: {close[-1]}, tp: {take_profit}, sl: {stop_loss} at time {start_time[-1]}"
                    )


config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=[],
    active_order_interval=1,
    initial_capital=10000.0,
    candle_topics=["candles-" + Strategy.time_interval + "-BTC/USDT-bybit"],
    start_time=datetime(2023, 6, 9, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 6, 10, 0, 0, 0, tzinfo=timezone.utc),
    api_key="YOUR_CYBOTRADE_API_KEY",
    api_secret="YOUR_CYBOTRADE_API_SECRET",
    data_count=500,
    # exchange_keys="./z_exchange-keys.json",
)

df = []

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["sma_1"] = [Strategy.sma_1]
hyper_parameters["sma_2"] = [Strategy.sma_2]
hyper_parameters["sl_low_length"] = [Strategy.sl_low_length]
hyper_parameters["sl_max_percentage"] = [Strategy.sl_max_percentage]
hyper_parameters["macd_crossline_lookback"] = [Strategy.macd_crossline_lookback]
hyper_parameters["risk_to_reward"] = [Strategy.risk_to_reward]
hyper_parameters["time_interval"] = [Strategy.time_interval]


async def start_backtest():
    await permutation.run(hyper_parameters, Strategy)


asyncio.run(start_backtest())

# Start of my own code stack, only using Cybotrade for data collection. BACKTEST ONLY
# backtest_table = pd.DataFrame(df)
# backtest_table.columns = ['start_time', 'low', 'close']

#Hyperparams
# sma_1 = 30
# sma_2 = 40
# macd_slow_line = 26
# macd_fast_line = 12
# macd_signal = 9
# sl_low_length = 10
# macd_crossline_lookback = 10
# risk_to_reward = 1.5
# positions = False
# macd_crossover_outer = False
#
# backtest_table['moving_average_30'] = talib.SMA(backtest_table['close'], sma_1)
# backtest_table['moving_average_40'] = talib.SMA(backtest_table['close'], sma_2)
# backtest_table['macd_line'] = talib.MACD(backtest_table['close'], macd_fast_line, macd_slow_line, macd_signal)[0]
# backtest_table['signal_line'] = talib.MACD(backtest_table['close'], macd_fast_line, macd_slow_line, macd_signal)[1]

# print(backtest_table)
