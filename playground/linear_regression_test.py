import sys
from datetime import datetime

from backtesting import Strategy, Backtest
from backtesting.lib import crossover, barssince, plot_heatmaps
import talib
import numpy as np
from private_indicators import pivot_low
import tools

class LinearRegressionStrategy(Strategy):
    time_period = 32
    sma_length = 42
    atr_length = 20
    sl_coeff = 1.06
    risk_reward = 2
    def init(self):
        try:
            self.linear_reg = self.I(talib.LINEARREG, self.data.Close, self.time_period)
            self.sma = self.I(talib.SMA, self.data.Close, self.sma_length)
            self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)

        except Exception as e:
            print(f"Error initializing indicators {e}")
            sys.exit()

    def next(self):
        try:
            if not self.position.is_long and self.linear_reg[-1] > self.data.Close[-1] > self.sma[-1]:
                atr_sl = self.sl_coeff * self.atr[-1]  # calculate sl coefficient through atr
                long_stop_loss = self.data.Close[-1].item() - atr_sl
                long_take_profit = self.data.Close[-1].item() + atr_sl * self.risk_reward
                self.buy(size=0.01, sl=long_stop_loss, tp=long_take_profit)

            elif not self.position.is_short and self.linear_reg[-1] < self.data.Close[-1] < self.sma[-1]:
                atr_sl = self.sl_coeff * self.atr[-1]  # calculate sl coefficient through atr
                short_stop_loss = self.data.Close[-1].item() + atr_sl
                short_take_profit = self.data.Close[-1].item() - atr_sl * self.risk_reward
                self.sell(size=0.01, sl=short_stop_loss, tp=short_take_profit)

            if self.position.is_long and self.data.Close[-1] > self.linear_reg[-1] and self.linear_reg[-2] > self.linear_reg:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] < self.linear_reg[-1]:
                self.position.close()

        except Exception as e:
            print(f"Error in next method: {e} at {self.data.index[-1]}")
            sys.exit()

try:
    df = tools.extract_candles_csv('BTC_2020-2024_15m.csv', datetime(2023, 1, 1), datetime(2024, 12, 28))
    print(df)

    factor = 100
    bt = Backtest(df, LinearRegressionStrategy, cash=1000000, margin=0.01, trade_on_close=True)

    # Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False, plot_drawdown=True)

    # Optimization
    stats, heatmap = bt.optimize(
        # risk_reward=[2 + 0.1 * i for i in range(50)],
        # pivot_backcandles=range(5, 10, 1),
        # time_period=range(2, 60, 2),
        # sma_length=range(2, 200, 5),
        # atr_length=range(2, 50, 2),
        # sl_coeff=[1.01 + 0.05 * i for i in range(10)],
        risk_reward=[1 + 0.1 * i for i in range(80)],
        # ema_closeness=[0.1 + 0.05 * i for i in range(20)],
        maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
        # constraint=lambda param: param.ema_2_length > param.ema_1_length,
        return_heatmap=True,
    )
    print(heatmap)
    print(stats)
    plot_heatmaps(heatmap, agg="mean")

    # # Show all the trades
    # trades = stats._trades
    #
    # trades.ReturnPct *= 100 # Converting 0.01 to 1%
    # print(trades.to_string())

except ValueError as e:
    print(f"Error: {e}")