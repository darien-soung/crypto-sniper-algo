from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from datetime import datetime
import tools
import talib

class KevinDavey(Strategy):
    ## This is an example of an ENTRY he uses, not a full strategy.

    atr_length = 14
    candle_1 = 40
    candle_2 = 50
    sl_coeff = 1.2
    risk_reward = 3

    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)

    def next(self):
        if len(self.data.Close) > self.candle_2 + 2:
            if (not self.position.is_long and
                    self.data.Close[-self.candle_1] < self.data.Close[-1] < self.data.Close[-self.candle_2] and
                    self.data.Low[-1] < self.data.Low[-2]):
                atr_sl = self.sl_coeff * self.atr[-1]
                long_stop_loss = self.data.Close[-1].item() - atr_sl
                long_take_profit = self.data.Close[-1].item() + atr_sl * self.risk_reward
                self.buy(size=0.01, sl=long_stop_loss, tp=long_take_profit)

            if (not self.position.is_short and
                    self.data.Close[-self.candle_1] > self.data.Close[-1] > self.data.Close[-self.candle_2] and
                    self.data.High[-1] > self.data.High[-2]):
                atr_sl = self.sl_coeff * self.atr[-1]
                short_stop_loss = self.data.Close[-1].item() + atr_sl
                short_take_profit = self.data.Close[-1].item() - atr_sl * self.risk_reward
                self.sell(size=0.01, sl=short_stop_loss, tp=short_take_profit)



try:
    df = tools.extract_candles_csv('BTC_2020-2024_1h.csv', datetime(2023, 1, 1), datetime(2024, 6, 30))
    print(df)
    # df.to_csv('test.csv')
    # Adjusting factors to the trade (because backtesting.py doesn't accept fractional shares)
    factor = 100

    bt = Backtest(df, KevinDavey, cash=1000000, margin=0.01) # 0.055%

    ## Backtest Execution
    # stats = bt.run()
    # print(stats)
    # bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False)

    ## Optimization
    stats, heatmap = bt.optimize(
        # atr_length=range(5, 20, 10),
        # risk_reward=[1 + 0.1 * i for i in range(50)],
        # sl_coeff=[1.01 + 0.05 * i for i in range(10)],
        candle_1=range(10, 100, 5),
        candle_2=range(20, 150, 5),
        maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
        constraint=lambda param: param.candle_2 > param.candle_1,
        return_heatmap=True,
    )
    print(stats)
    print(heatmap)
    plot_heatmaps(heatmap, agg="mean")

except ValueError as e:
    print(f"Error: {e}")