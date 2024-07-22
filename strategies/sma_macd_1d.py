import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime, timedelta
import talib
import tools
from private_indicators import crossover, crossover_zero


class SmaMacdCross(Strategy):
    # Define the two MA length
    length_1 = 30
    length_2 = 40
    macd_cross_line_lookback = 5
    sl_low_length = 10
    sl_max_percentage = 0.05
    risk_to_reward = 3
    damage_cost = 0.05  # 5%

    def init(self):
        self.sma_1 = self.I(talib.SMA, self.data.Close, self.length_1)
        self.sma_2 = self.I(talib.SMA, self.data.Close, self.length_2)
        self.macd = self.I(talib.MACD, self.data.Close, 12, 26, 9)

    def next(self):
        price = self.data.Close[-1]
        remaining_cash = self.equity
        date = self.data.index[-1]
        macd_line = self.macd[0]
        signal_line = self.macd[1]

        if not self.position:  # Ensure no strategies are open
            if crossover(self.sma_1, self.sma_2) > 0 and self.sma_1[-1] < self.data.Close[-1]:  # Crossover to the upside and price is above the moving average
                # Check for MACD crossover the 0 line in the last n data points
                # if crossover_zero(self.macd[0], self.macd_cross_line_lookback) > 0:
                #     or_stop_loss = stop_loss = min(
                #         self.data.Low[len(self.data.Low) - self.sl_low_length: len(self.data.Low)])
                #     if (self.data.Close[-1] - stop_loss) / self.data.Close[-1] > self.sl_max_percentage:
                #         stop_loss = self.data.Close[-1] * (1 - self.sl_max_percentage)
                #     take_profit = self.data.Close[-1] + (self.risk_to_reward * (self.data.Close[-1] - or_stop_loss))
                #     self.buy(size=0.5, sl=stop_loss, tp=take_profit)
                for i in range(np.negative(self.macd_cross_line_lookback)+1, 0):
                    if macd_line[i] > 0 >= macd_line[i-1]:
                        or_stop_loss = stop_loss = min(self.data.Low[len(self.data.Low) - self.sl_low_length: len(self.data.Low)])
                        if tools.find_percentage(self.data.Close[-1], stop_loss) > self.sl_max_percentage:
                            stop_loss = self.data.Close[-1] * (1 - self.sl_max_percentage)
                        take_profit = self.data.Close[-1] + (self.risk_to_reward * (self.data.Close[-1] - or_stop_loss))
                        trade_size = ((remaining_cash * self.damage_cost /
                                      (tools.find_percentage(self.data.Close[-1], stop_loss) * 100)) /
                                      remaining_cash) # Used to specify risk per trade
                        self.buy(size=0.03, sl=stop_loss, tp=take_profit)
                        break
                    


try:
    df = tools.extract_candles_binance(datetime(2023, 6, 22), datetime(2024, 6, 24), "1h", 'BTCUSDT')
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100
    # columns_to_scale = ['Open', 'High', 'Low', 'Close']
    # df[columns_to_scale] = df[columns_to_scale] / factor

    print(df)

    bt = Backtest(df, SmaMacdCross, cash=1000000, margin=0.01)

    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True)

    # # Optimization
    # stats, heatmap = bt.optimize(
    #     length_1=range(30, 100, 5),
    #     length_2=range(40,150, 5),
    #     risk_to_reward=range(1, 4, 1),
    #     maximize='Sharpe Ratio',
    #     constraint=lambda param: param.length_2 > param.length_1,
    #     return_heatmap=True
    #
    # )
    # print(heatmap)
    # plot_heatmaps(heatmap, agg="mean")


    # # Show all the trades
    # trades = stats._trades

    ## Scale for cash adjustment
    # trades.Size /= factor
    # trades.PnL /= factor

    # trades.ReturnPct *= 100 # Converting 0.01 to 1%
    # print(trades.to_string())


except ValueError as e:
    print(f"Error: {e}")


