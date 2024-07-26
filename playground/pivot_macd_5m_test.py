import sys

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime
from private_indicators import crossover, pivot_low, pivot_high, is_price_away, is_price_within
import talib
import tools


class PivotMacdStrategy(Strategy):
    # define parameters
    pivot_backcandles = 18
    risk_reward = 1.5
    macd_backcandles = 6

    def init(self):
        try:
            self.macd = self.I(talib.MACD, self.data.Close, 12, 26, 9)
            self.pivot_lows = self.I(pivot_low, self.data.Low, self.pivot_backcandles, scatter=True, overlay=True) # #FF0000
            self.pivot_highs = self.I(pivot_high, self.data.High, self.pivot_backcandles, scatter=True, overlay=True) # #00FF00

        except Exception as e:
            print(f"Error initializing indicators: {e}")
            sys.exit()

    def next(self):
        try:
            macd_fast = self.macd[0]
            macd_slow = self.macd[1]

            ## For debug, clear after no bug confirmed.
            # print(f"Pivot high: {self.pivot_highs[-1]}")
            # print(f"Pivot lows: {self.pivot_lows[-1]}")
            # print(f"MACD fast: {macd_fast[-1]}")
            # print(f"MACD slow: {macd_slow[-1]}")
            # current_price = self.data.Close[-1].item()
            # current_high = self.data.High[-1].item()
            # current_pivot_signal_high = self.pivot_highs[-1].item()
            # current_datetime = self.data.index[-1]

            pivot_low_detected = False
            pivot_high_detected = False
            pivot_backcandles_index = -(self.pivot_backcandles * 2 + 1)
            middle_index = pivot_backcandles_index // 2

            if self.data.Low[middle_index] == min(self.data.Low[pivot_backcandles_index:]):
                pivot_low_detected = True
            if self.data.High[middle_index] == max(self.data.High[pivot_backcandles_index:]):
                pivot_high_detected = True



            if pivot_low_detected and not pivot_high_detected and not self.position:
                if is_price_away(self.data.Close[-1], self.data.Low[middle_index].item(), 0.1):
                    for i in range(1, self.macd_backcandles + 1):
                        if crossover(macd_fast[:-i], macd_slow[:-i]) > 0 > macd_slow[-1]:
                            stop_loss = self.data.Low[middle_index].item()
                            take_profit = self.data.Close[-1] + (self.risk_reward * (self.data.Close[-1] - stop_loss))
                            self.buy(size=0.01, sl=stop_loss, tp=take_profit)
                            break

            elif pivot_high_detected and not pivot_low_detected and not self.position:
                if is_price_away(self.data.Close[-1], self.data.High[middle_index].item(), 0.1):
                    for i in range(1, self.macd_backcandles + 1):
                        if crossover(macd_fast[:-i], macd_slow[:-i]) < 0 < macd_slow[-1]:
                            stop_loss = self.data.High[middle_index].item()
                            take_profit = self.data.Close[-1] - (self.risk_reward * (stop_loss - self.data.Close[-1]))
                            self.sell(size=0.01, sl=stop_loss, tp=take_profit)
                            break

            # if pivot_low_detected and not pivot_high_detected and not self.position:
            #     if is_price_away(self.data.Close[-1], self.data.Low[middle_index].item(), 0.1):
            #         # if self.rsi[-1] < self.lower_threshold_rsi:
            #         stop_loss = self.data.Low[middle_index].item()
            #         take_profit = self.data.Close[-1] + (self.risk_reward * (self.data.Close[-1] - stop_loss))
            #         self.buy(size=0.01, sl=stop_loss, tp=take_profit)
            #
            # elif pivot_high_detected and not pivot_low_detected and not self.position:
            #     if is_price_away(self.data.Close[-1], self.data.High[middle_index].item(), 0.1):
            #         # if self.rsi[-1] > self.upper_threshold_rsi:
            #         stop_loss = self.data.High[middle_index].item()
            #         take_profit = self.data.Close[-1] - (self.risk_reward * (stop_loss - self.data.Close[-1]))
            #         self.sell(size=0.01, sl=stop_loss, tp=take_profit)


        except Exception as e:
            print(f"Error in next method: {e} at {self.data.index[-1]}")
            sys.exit()


try:
    df = tools.extract_candles_csv('BTC_2020-2024_5m.csv', datetime(2023, 1, 22), datetime(2024, 6, 28))
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    bt = Backtest(df, PivotMacdStrategy, cash=1000000, margin=0.01)

    ## Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False)

    ## Optimization
    # stats, heatmap = bt.optimize(
    #     # risk_reward=[1 + 0.1 * i for i in range(40)],
    #     pivot_backcandles=range(10, 26, 1),
    #     macd_backcandles=range(1, 10, 1),
    #     maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
    #     # constraint=lambda param: param.ema_2_length > param.ema_1_length,
    #     return_heatmap=True,
    # )
    # print(heatmap)
    # print(stats)
    # plot_heatmaps(heatmap, agg="mean")

except ValueError as e:
    print(f"Error: {e}")