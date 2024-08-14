import sys
from datetime import datetime

from backtesting import Strategy, Backtest
from backtesting.lib import crossover, barssince, plot_heatmaps
import talib
import numpy as np
from private_indicators import pivot_low, is_price_away
import tools


class MACDDivergenceStrategy(Strategy):
    pivot_backcandles = 5
    MAX_RANGE = 50
    MIN_RANGE = 5
    risk_reward = 3
    bollinger_length = 40
    bollinger_stddev = 2.5
    minimum_band_distance = 1.35
    atr_length = 14
    sl_coeff = 2

    def init(self):
        try:
            # Calculate indicators
            self.macd = self.I(talib.MACD, self.data.Close, 12, 26, 9)
            self.pivot_lows = self.I(pivot_low, self.data.Low, self.pivot_backcandles, scatter=True,
                                     overlay=True)  # #FF0000
            self.macd_pivot_lows = self.I(pivot_low, self.macd, self.pivot_backcandles, scatter=True)  # #FF0000
            # self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)
            self.bollinger = self.I(talib.BBANDS, self.data.Close, self.bollinger_length, self.bollinger_stddev,
                                    self.bollinger_stddev, 0)

        except Exception as e:
            print(f"Error initializing indicators {e}")

    def next(self):
        try:
            upper_band = self.bollinger[0]
            middle_band = self.bollinger[1]
            lower_band = self.bollinger[2]
            macd_line = self.macd[0]
            signal_line = self.macd[1]
            pivot_backcandles_index = -(self.pivot_backcandles * 2 + 1)  # Will always be -ve
            middle_index = pivot_backcandles_index // 2  # Will always be -ve

            if macd_line[middle_index] == min(macd_line[pivot_backcandles_index:]) and \
                    len(self.data.Close) > self.MAX_RANGE - pivot_backcandles_index:
                # Find second pivot low in MACD within the min and max range
                if not self.position:
                    for i in range(pivot_backcandles_index - self.MIN_RANGE, pivot_backcandles_index - self.MAX_RANGE + 5,
                                   -1):  # To try: find divergences from the back of the range FIRST
                        if macd_line[i] == min(macd_line[i - 5:i + 5]):
                            # MACD higher low check and Price lower low check
                            if macd_line[middle_index] > macd_line[i] and self.data.Low[middle_index] < self.data.Low[i] and \
                                    self.data.Low[middle_index] < self.data.Close[-1]:  # Price has to be lower than MACD divergence

                                # print(f"Long Trade initiated at {self.data.Close[-1]} at {self.data.index[-1]} found bull div at Price {self.data.Low[i]} > {self.data.Low[middle_index]} respectively at {self.data.index[i]} and {self.data.index[middle_index]}")

                                stop_loss = self.data.Low[middle_index].item()
                                take_profit = self.data.Close[-1] + (self.risk_reward * (self.data.Close[-1] - stop_loss))
                                self.buy(size=0.012, sl=stop_loss, tp=take_profit)

                            break

            if macd_line[middle_index] == max(macd_line[pivot_backcandles_index:]) and \
                    len(self.data.Close) > self.MAX_RANGE - pivot_backcandles_index:
                # Find second pivot low in MACD within the min and max range
                if not self.position:
                    for i in range(pivot_backcandles_index - self.MIN_RANGE, pivot_backcandles_index - self.MAX_RANGE + 5,
                                   -1):  # To try: find divergences from the back of the range FIRST
                        if macd_line[i] == max(macd_line[i - 5:i + 5]):
                            # Price higher high check and MACD higher low check
                            if macd_line[middle_index] < macd_line[i] and self.data.High[middle_index] > self.data.High[i] and \
                                    self.data.High[middle_index] > self.data.Close[-1]:  # SL Price has to be higher than current price
                                # print(f"Shot Trade initiated at {self.data.Close[-1]} at {self.data.index[-1]} found bull div at Price {self.data.Low[i]} < {self.data.Low[middle_index]} respectively at {self.data.index[i]} and {self.data.index[middle_index]}")
                                stop_loss = self.data.High[middle_index].item()
                                take_profit = self.data.Close[-1] - (self.risk_reward * (stop_loss - self.data.Close[-1]))
                                self.sell(size=0.012, sl=stop_loss, tp=take_profit)

                            break


        except Exception as e:
            print(f"Error in next method: {e} at {self.data.index[-1]}")
            sys.exit()


try:
    df = tools.extract_candles_csv('BTC_2020-2024_4h.csv', datetime(2020, 1, 1), datetime(2024, 12, 28))
    print(df)

    factor = 100
    bt = Backtest(df, MACDDivergenceStrategy, cash=1000000, margin=0.01, trade_on_close=True)

    # Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False, plot_drawdown=True)

    # Optimization
    stats, heatmap = bt.optimize(
        # risk_reward=[2 + 0.1 * i for i in range(50)],
        # pivot_backcandles=range(5, 10, 1),
        bollinger_length=range(5, 50, 2),
        # bollinger_stddev=[1 + 0.1 * i for i in range(50)],
        minimum_band_distance=[0.05 + 0.05 * i for i in range(50)],
        # atr_lookback=range(2, 5, 1),
        # ema_closeness=[0.1 + 0.05 * i for i in range(20)],
        maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
        # constraint=lambda param: param.ema_2_length > param.ema_1_length,
        return_heatmap=True,
    )
    print(heatmap)
    print(stats)
    plot_heatmaps(heatmap, agg="mean")

    # Show all the trades
    trades = stats._trades

    trades.ReturnPct *= 100 # Converting 0.01 to 1%
    print(trades.to_string())

except ValueError as e:
    print(f"Error: {e}")
