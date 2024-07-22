import sys

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime, timedelta
from private_indicators import zscore, crossover
import talib
import tools


class ScalpVwapRsiTest(Strategy): # Basically a mean reversion strategy that theoretically works well in ranging environments.
    # define parameters
    ema_1_length = 30
    ema_2_length = 50
    rsi_length = 10
    bollinger_length = 20
    bollinger_stddev = 2
    atr_length = 14
    sl_coeff = 1.4
    risk_reward = 1.5
    ema_backcandles = 7
    vwap_backcandles = 15

    def init(self):
        try:
            # self.ema_fast = self.I(talib.EMA, self.data.Close, self.ema_1_length)
            # self.ema_slow = self.I(talib.EMA, self.data.Close, self.ema_2_length)
            self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_length)
            self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close,self.data.Volume)
            self.bollinger = self.I(talib.BBANDS, self.data.Close, self.bollinger_length, self.bollinger_stddev,
                                    self.bollinger_stddev, 0)
            self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)

        except Exception as e:
            print(f"Error initializing indicators: {e}")
            sys.exit()

    def next(self):
        # remaining_cash = self.equity
        upper_band = self.bollinger[0]
        lower_band = self.bollinger[2]
        uptrend = 1
        downtrend = 1
        overall_trend = 0

        for i in range(-self.vwap_backcandles, 0):
            if max(self.data.Open[i].item(), self.data.Close[i].item()) > self.vwap[i]:
                uptrend = 0
            elif min(self.data.Open[i].item(), self.data.Close[i].item()) < self.vwap[i]:
                downtrend = 0

            if uptrend == 1 and downtrend == 1:
                overall_trend = 0
            elif uptrend == 1:
                overall_trend = 1
            elif downtrend == 1:
                overall_trend = -1

        try:
            if (overall_trend == 1 and # Uptrend
                    self.data.Close[-1] <= lower_band[-1] and
                    self.rsi[-1] < 45 and
                    not self.position.is_long): # Price pierces through the lower band
                atr_sl = self.sl_coeff * self.atr[-1] # calculate sl coefficient through atr
                long_stop_loss = self.data.Close[-1].item() - atr_sl
                long_take_profit = self.data.Close[-1].item() + atr_sl * self.risk_reward

                self.buy(size=0.01, sl=long_stop_loss, tp=long_take_profit)


            elif (overall_trend == -1 and # Downtrend
                    self.data.Close[-1] >= upper_band[-1] and
                    self.rsi[-1]>55 and
                    not self.position.is_short): # Price pierces through the upper band
                atr_sl = self.sl_coeff * self.atr[-1] # calculate sl coefficient through atr
                short_stop_loss = self.data.Close[-1].item() + atr_sl
                short_take_profit = self.data.Close[-1].item() - atr_sl * self.risk_reward

                self.sell(size=0.01, sl=short_stop_loss, tp=short_take_profit)

            if len(self.trades) > 0:
                if self.trades[-1].is_long and self.rsi[-1] >= 90:
                    self.trades[-1].close()
                elif self.trades[-1].is_short and self.rsi[-1] <= 10:
                    self.trades[-1].close()


        except Exception as e:
            print(f"Error in next method: {e}")
            sys.exit()


try:
    df = tools.extract_candles_csv('BTC_2020-2024_5m.csv', datetime(2024, 5, 22), datetime(2024, 6, 24))
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    bt = Backtest(df, ScalpVwapRsiTest, cash=1000000, margin=0.01)

    ## Backtest Execution
    # stats = bt.run()
    # print(stats)
    # bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False)

    ## Optimization
    stats, heatmap = bt.optimize(
        atr_length=range(5, 20, 1),
        risk_reward=[1 + 0.1 * i for i in range(10)],
        sl_coeff=[1.01 + 0.05 * i for i in range(10)],
        # bollinger_length=range(5, 20, 1),
        # bollinger_stddev=[0.7 + 0.1 * i for i in range(30)],
        maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
        # constraint=lambda param: param.ema_2_length > param.ema_1_length,
        return_heatmap=True,
    )
    print(heatmap)
    plot_heatmaps(heatmap, agg="mean")

except ValueError as e:
    print(f"Error: {e}")
