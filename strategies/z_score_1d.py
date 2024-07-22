import sys

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime, timedelta
from private_indicators import zscore, crossover
import talib
import tools

class zScoreStrategy(Strategy):
    # define parameters
    window = 75
    z_score_threshold = 0.75

    def init(self):
        try:
            self.sma_1 = self.I(talib.SMA, self.data.Close, self.window)
            self.zscore = self.I(zscore, self.data.Close, self.window)
        except Exception as e:
            print(f"Error initializing indicators: {e}")
            sys.exit()

    def next(self):
        remaining_cash = self.equity

        try:
            if self.zscore is not None and len(self.zscore) > 0:
                if not self.position:
                    # print(f"{self.data.index[-1]}: zscore: {self.zscore[-1]}")
                    if crossover(self.zscore, self.z_score_threshold) > 0:
                        stop_loss = self.data.Close[-1].item() * 0.94
                        self.buy(size=0.02)
                else:
                    if crossover(self.zscore, self.z_score_threshold) < 0: # or self.zscore > 3.5:
                        self.position.close()
        except Exception as e:
            print(f"Error in next method: {e}")
            sys.exit()

try:
    df = tools.extract_candles_binance(datetime(2020, 6, 22), datetime(2024, 6, 24), "1d", 'BTCUSDT')
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    print(df)

    # Random git push change made

    bt = Backtest(df, zScoreStrategy, cash=1000000, margin=0.01)

    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True)

    # Optimization
    # stats, heatmap = bt.optimize(
    #     window=range(20, 100, 1),
    #     z_score_threshold= [0.3 + 0.05 * i for i in range(15)], # [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
    #     maximize='Return [%]', # Max. Drawdown [%], Return [%], Sharpe Ratio
    #     return_heatmap=True,
    # )
    # print(heatmap)
    # plot_heatmaps(heatmap, agg="mean")

    # # Show all the trades
    # trades = stats._trades
    #
    # # Scale for cash adjustment
    # trades.Size /= factor
    # trades.PnL /= factor
    #
    # trades.ReturnPct *= 100 # Converting 0.01 to 1%
    # print(trades.to_string())


except ValueError as e:
    print(f"Error: {e}")