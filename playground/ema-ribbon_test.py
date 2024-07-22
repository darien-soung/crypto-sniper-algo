from backtesting import Strategy, Backtest
from backtesting.lib import plot_heatmaps
from datetime import datetime
import talib
import tools
from private_indicators import crossover, is_price_within

class EmaRibbonStrategy(Strategy):
    # define parameters
    ema_1_l = 5
    ema_2_l = 8
    ema_3_l = 13
    ema_4_l = 20
    ema_5_l = 50
    risk_reward = 5
    compressed_backcandles = 3
    ema_closeness = 0.3

    def init(self):
        try:
            self.ema_1 = self.I(talib.EMA, self.data.Close, self.ema_1_l, color="#FF0000")
            self.ema_2 = self.I(talib.EMA, self.data.Close, self.ema_2_l, color="#FFFF00")
            self.ema_3 = self.I(talib.EMA, self.data.Close, self.ema_3_l, color="#0000FF")
            self.ema_4 = self.I(talib.EMA, self.data.Close, self.ema_4_l, color="#00FF00")
            self.ema_5 = self.I(talib.EMA, self.data.Close, self.ema_5_l, color="#A020F0")

        except Exception as e:
            print(f"Error initializing indicators {e}")


    def next(self):
        current_datetime = self.data.index[-1]
        ema_values = [self.ema_1[-1].item(), self.ema_2[-1].item(), self.ema_3[-1].item(), self.ema_4[-1].item(),
                      self.ema_5[-1].item()]

        if not self.position:
            if self.data.Close[-1] > self.ema_1[-1] > self.data.Low[-1]: # price crosses ema
                if (crossover(self.ema_1, self.ema_5) and # fast ema crosses slow ema
                        any(ema_values[0] >= ema for ema in ema_values[1:])):
                    for i in range(-self.compressed_backcandles, 0): # compressed backcandles
                        ema_values_i = [self.ema_1[i].item(), self.ema_2[i].item(), self.ema_3[i].item(), self.ema_4[i].item(), self.ema_5[i].item()]
                        if is_price_within(max(ema_values_i), min(ema_values_i), 0.3):
                            stop_loss = self.data.Low[-1].item()
                            take_profit = self.data.Close[-1] - (self.risk_reward * (stop_loss - self.data.Close[-1]))
                            self.buy(size=75, sl=stop_loss, tp=take_profit)
                            break


try:
    df = tools.extract_candles_csv('BTC_2020-2024_4h.csv', datetime(2023, 1, 1), datetime(2024, 6, 28))
    print(df)

    factor = 100
    bt = Backtest(df, EmaRibbonStrategy, cash=1000000, margin=0.01, trade_on_close=True)

    ## Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=True, plot_drawdown=True)

    ## Optimization
    # stats, heatmap = bt.optimize(
    #     # risk_reward=[2 + 0.1 * i for i in range(50)],
    #     compressed_backcandles=range(1, 8, 1),
    #     ema_closeness=[0.1 + 0.05 * i for i in range(20)],
    #     maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
    #     # constraint=lambda param: param.ema_2_length > param.ema_1_length,
    #     return_heatmap=True,
    # )
    # print(heatmap)
    # print(stats)
    # plot_heatmaps(heatmap, agg="mean")

except ValueError as e:
    print(f"Error: {e}")



