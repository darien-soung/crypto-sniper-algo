import sys

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime, timedelta
from private_indicators import zscore, crossover
import talib
import tools


class z_score_15min(Strategy):
    # define parameters
    window = 95
    z_score_threshold = 1

    def init(self):
        try:
            self.sma_1 = self.I(talib.SMA, self.data.Close, 200)
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
                        self.buy(size=0.02)
                else:
                    if crossover(self.zscore, self.z_score_threshold / 2) < 0:  # or self.zscore > 3.5:
                        self.position.close()
        except Exception as e:
            print(f"Error in next method: {e}")
            sys.exit()


try:
    df = tools.extract_candles_binance(datetime(2023, 6, 22), datetime(2024, 6, 24), "1h", 'BTCUSDT')
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    bt = Backtest(df, z_score_15min, cash=1000000, margin=0.01)

    # Backtest Execution
    # stats = bt.run()
    # print(stats)
    # bt.plot(plot_volume=False, superimpose=False, open_browser=True)

    # Optimization
    stats, heatmap = bt.optimize(
        window=range(20, 100, 2),
        z_score_threshold=[0.6 + 0.05 * i for i in range(20)],  # [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
        maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
        return_heatmap=True,
    )
    print(heatmap)
    plot_heatmaps(heatmap, agg="mean")

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
