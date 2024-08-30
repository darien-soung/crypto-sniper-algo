import sys

from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from datetime import datetime
from private_indicators import trend_akkam, williams_percent_r, coppock_curve, atr_bands
import talib
import tools


class AkkamStrategy(Strategy):
    risk_reward = 1.8
    atr_length = 14
    williams_length = 7
    coppock_length = 10

    def init(self):
        try:
            self.williams = self.I(williams_percent_r, self.data.High, self.data.Low, self.data.Close, self.williams_length, 50)
            self.akkam = self.I(trend_akkam, self.data.Open, self.data.High, self.data.Low, self.data.Close, True,
                                 False,
                                 False, 50, 6, 10.0, 0, 100)
            self.coppock = self.I(coppock_curve, self.data.Close, self.coppock_length)
            self.atr_bands = self.I(atr_bands, self.data.High, self.data.Low, self.data.Close, self.atr_length, 3.0, 3.0)


        except Exception as e:
            print(f"Error initializing indicators: {e}")
            sys.exit()

    def next(self):
        try:
            test = 0

        except Exception as e:
            print(f"Error in next method: {e} at {self.data.index[-1]}")
            sys.exit()

try:
    df = tools.extract_candles_csv('BTC_2020-2024_1d.csv', datetime(2023, 1, 1), datetime(2024, 12, 30))
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    bt = Backtest(df, AkkamStrategy, cash=1000000, margin=0.01) # 0.055%

    ## Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=True, filename="./performance/akkam_p")

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
    # plot_heatmaps(heatmap, agg="mean", filename="./optimization/akkam_o")

except ValueError as e:
    print(f"Error: {e}")