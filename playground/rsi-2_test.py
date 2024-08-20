import sys
from datetime import datetime

from backtesting import Strategy, Backtest
from backtesting.lib import crossover, barssince, plot_heatmaps
import talib
import numpy as np
from private_indicators import pivot_low, is_price_away
import tools


class RSI2(Strategy):
    RSI_PERIOD = 2
    sma_length_slow = 200
    sma_length_fast = 5
    atr_length = 14
    sl_coeff = 1.2
    risk_reward = 2


    def init(self):
        try:
            # Calculate RSI
            self.rsi = self.I(talib.RSI, self.data.Close, self.RSI_PERIOD)
            self.sma_200 = self.I(talib.SMA, self.data.Close, self.sma_length_slow)
            self.sma_5 = self.I(talib.SMA, self.data.Close, self.sma_length_fast)
            self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)


        except Exception as e:
            print(f"Error initializing indicators {e}")

    def next(self):
        try:
            if not self.position and self.rsi[-1] < 10 and self.data.Close[-1] > self.sma_200 and self.data.Low[-1] != self.data.Close[-1]:
                atr_sl = self.sl_coeff * self.atr[-1]  # calculate sl coefficient through atr
                stop_loss = self.data.Close[-1].item() - atr_sl
                take_profit = self.data.Close[-1].item() + atr_sl * self.risk_reward
                # stop_loss = self.data.Low[-1].item()
                # take_profit = self.data.Close[-1] - (self.risk_reward * (stop_loss - self.data.Close[-1]))
                self.buy(size=0.01, sl=stop_loss, tp=take_profit)

            elif not self.position and self.rsi[-1] > 90 and self.data.Close[-1] < self.sma_200 and self.data.High[-1] != self.data.Close[-1]:
                atr_sl = self.sl_coeff * self.atr[-1]  # calculate sl coefficient through atr
                stop_loss = self.data.Close[-1].item() + atr_sl
                take_profit = self.data.Close[-1].item() - atr_sl * self.risk_reward
                # stop_loss = self.data.High[-1].item()
                # take_profit = self.data.Close[-1] - (self.risk_reward * (stop_loss - self.data.Close[-1]))

                self.sell(size=0.01, sl=stop_loss, tp=take_profit)


        except Exception as e:
            print(f"Error in next method: {e} at {self.data.index[-1]}")
            sys.exit()


try:
    df = tools.extract_candles_csv('BTC_2020-2024_1d.csv', datetime(2023, 1, 1), datetime(2024, 12, 28))
    print(df)

    factor = 100
    bt = Backtest(df, RSI2, cash=1000000, margin=0.01, trade_on_close=True)

    # Backtest Execution
    # stats = bt.run()
    # print(stats)
    # bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False, plot_drawdown=True)

    # Optimization
    stats, heatmap = bt.optimize(
        risk_reward=[2 + 0.1 * i for i in range(50)],
        sl_coeff=[1.01 + 0.05 * i for i in range(10)],
        maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
        # constraint=lambda param: param.ema_2_length > param.ema_1_length,
        return_heatmap=True,
    )
    print(heatmap)
    print(stats)
    plot_heatmaps(heatmap, agg="mean")

    # Show all the trades
    # trades = stats._trades
    #
    # trades.ReturnPct *= 100 # Converting 0.01 to 1%
    # print(trades.to_string())

except ValueError as e:
    print(f"Error: {e}")
