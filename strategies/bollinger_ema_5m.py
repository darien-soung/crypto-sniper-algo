import sys

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime, timedelta
from private_indicators import is_price_away
import talib
import tools


class BollingerEMAStrategy(Strategy): # Basically a mean reversion strategy that theoretically works well in ranging environments.
    # define parameters
    ema_1_length = 30
    ema_2_length = 50
    rsi_length = 10
    bollinger_length = 17
    bollinger_stddev = 2.5
    atr_length = 15
    sl_coeff = 1.41
    risk_reward = 2
    ema_backcandles = 7
    minimum_band_distance = 0.55 # 2 # for inverse

    def init(self):
        try:
            self.ema_fast = self.I(talib.EMA, self.data.Close, self.ema_1_length)
            self.ema_slow = self.I(talib.EMA, self.data.Close, self.ema_2_length)
            # self.sma = self.I(talib.SMA, self.data.Close, self.sma_length)
            self.bollinger = self.I(talib.BBANDS, self.data.Close, self.bollinger_length, self.bollinger_stddev,
                                    self.bollinger_stddev, 0)
            # self.bollinger_volatility = self.I(calculate_bollinger_volatility, self.bollinger[0], self.bollinger[2], self.bollinger_length, self.sma)
            self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)

        except Exception as e:
            print(f"Error initializing indicators: {e}")
            sys.exit()

    def next(self):
        # remaining_cash = self.equity
        upper_band = self.bollinger[0]
        lower_band = self.bollinger[2]

        try:

            if (all(self.ema_fast[-self.ema_backcandles:] > self.ema_slow[-self.ema_backcandles:]) and # Uptrend
                    self.data.Close[-1] <= lower_band[-1] and
                    is_price_away(upper_band, lower_band, self.minimum_band_distance) and
                    # self.bollinger_volatility[-1] < 1 and
                    not self.position.is_long): # Price pierces through the lower band
                atr_sl = self.sl_coeff * self.atr[-1] # calculate sl coefficient through atr
                long_stop_loss = self.data.Close[-1].item() - atr_sl
                long_take_profit = self.data.Close[-1].item() + atr_sl * self.risk_reward

                self.buy(size=0.015, sl=long_stop_loss, tp=long_take_profit)


            elif (all(self.ema_fast[-self.ema_backcandles:] < self.ema_slow[-self.ema_backcandles:]) and # Downtrend
                    self.data.Close[-1] >= upper_band[-1] and
                    is_price_away(upper_band, lower_band, self.minimum_band_distance) and
                    # self.bollinger_volatility[-1] < 1 and
                    not self.position.is_short): # Price pierces through the upper band
                atr_sl = self.sl_coeff * self.atr[-1] # calculate sl coefficient through atr
                short_stop_loss = self.data.Close[-1].item() + atr_sl
                short_take_profit = self.data.Close[-1].item() - atr_sl * self.risk_reward

                self.sell(size=0.015, sl=short_stop_loss, tp=short_take_profit)
        except Exception as e:
            print(f"Error in next method: {e}")
            sys.exit()


try:
    df = tools.extract_candles_csv('BTC_2020-2024_5m.csv', datetime(2023, 1, 1), datetime(2023, 1, 30))
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    bt = Backtest(df, BollingerEMAStrategy, cash=1000000, margin=0.01, commission=0.00055) # 0.055%

    ## Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False)

    ## Optimization
    # stats, heatmap = bt.optimize(
    #     atr_length=range(5, 20, 10),
    #     # risk_reward=[1 + 0.1 * i for i in range(10)],
    #     # sl_coeff=[1.01 + 0.05 * i for i in range(10)],
    #     minimum_band_distance=[0.05 + 0.05 * i for i in range(50)],
    #     # bollinger_length=range(5, 20, 1),
    #     # bollinger_stddev=[0.7 + 0.1 * i for i in range(30)],
    #     maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
    #     # constraint=lambda param: param.ema_2_length > param.ema_1_length,
    #     return_heatmap=True,
    # )
    # print(stats)
    # print(heatmap)
    # plot_heatmaps(heatmap, agg="mean")

except ValueError as e:
    print(f"Error: {e}")
