from backtesting import Backtest, Strategy
from datetime import datetime

from private_indicators import calculate_realized_volatility
import tools

class VolatilityTest(Strategy):
    rolling_window = 25

    def init(self):
        self.volatility = self.I(calculate_realized_volatility, self.data.High, self.data.Low, self.data.Close, self.rolling_window)


    def next(self):
        if not self.position:
            self.buy()


try:
    df = tools.extract_candles_csv('BTC_2020-2024_15m.csv', datetime(2023, 1, 1), datetime(2024, 12, 28))
    print(df)

    factor = 100
    bt = Backtest(df, VolatilityTest, cash=1000000, margin=0.01, trade_on_close=True)

    # Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=False, superimpose=False, open_browser=True, resample=False, plot_drawdown=True)

    # Optimization
    # stats, heatmap = bt.optimize(
    #     # risk_reward=[2 + 0.1 * i for i in range(50)],
    #     # pivot_backcandles=range(5, 10, 1),
    #     # RSI_PERIOD=range(5, 60, 2),
    #     # atr_length=range(100, 500, 2),
    #     # atr_limit=range(70, 71, 1),
    #     MAX_RANGE=range(20, 80, 1),
    #     MIN_RANGE=range(3, 15, 1),
    #     # ema_closeness=[0.1 + 0.05 * i for i in range(20)],
    #     maximize='Sharpe Ratio',  # Max. Drawdown [%], Return [%], Sharpe Ratio
    #     # constraint=lambda param: param.ema_2_length > param.ema_1_length,
    #     return_heatmap=True,
    # )
    # print(heatmap)
    # print(stats)
    # plot_heatmaps(heatmap, agg="mean")

    # Show all the trades
    # trades = stats._trades
    #
    # trades.ReturnPct *= 100 # Converting 0.01 to 1%
    # print(trades.to_string())

except ValueError as e:
    print(f"Error: {e}")
