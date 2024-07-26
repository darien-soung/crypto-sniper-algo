import sys

from backtesting import Strategy, Backtest
from backtesting.lib import plot_heatmaps
from datetime import datetime, time
import talib
import tools
from private_indicators import session_volume_profile



class SessionVolumeStrategy(Strategy):
    """
    THIS STRATEGY IS WRITTEN SPECIFICALLY FOR 1H CHART. DO NOT CHANGE TIMEFRAMES OR IT'LL NOT WORK
    """
    lookback = 24

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.previous_day_vah = 0
        self.close_above_vah_day = False
        self.previous_day_val = 0
        self.close_below_val_day = False
        self.position_opened_on_day = False

    def init(self):
        try:
            self.svp = self.I(session_volume_profile, self.data.index, self.data.High,
                                                  self.data.Low, self.data.Volume, self.lookback, overlay=True)

        except Exception as e:
            print(f"Error initializing indicators {e}")

    def next(self):
        try:
            current_datetime = self.data.index[-1]

            if current_datetime == datetime(2024, 5, 19, 0, 0, 0):
                ball = True

            # Initialize values for trade entries
            if current_datetime.time() == time(0, 0):
                if self.data.Open[-1] < self.svp[2][-2]:  # If Open is below previous day VAL
                    self.close_below_val_day = True
                    self.previous_day_val = self.svp[2][-2]
                    self.previous_day_vah = self.svp[0][-2]

                elif self.data.Open[-1] > self.svp[0][-2]: # If Open is above previous day VAH
                    self.close_above_vah_day = True
                    self.previous_day_vah = self.svp[0][-2]
                    self.previous_day_val = self.svp[2][-2]

            # Reset values at the end of the day for the next day ########################### To try: Set this logic at the start of 00:00, see if it works better
            elif current_datetime.time() == time(23, 0):
                self.previous_day_vah = 0
                self.close_above_vah_day = False
                self.previous_day_val = 0
                self.close_below_val_day = False
                self.position_opened_on_day = False

                if self.position:
                    self.position.close() # Close position if no TP or SL that day

            # Executes at any time between 00:00 and 23:00
            else:
                if self.close_below_val_day and not self.position and self.position_opened_on_day is False:
                    if self.data.Close[-1] >= self.previous_day_val: # If price closes back above val
                        self.position_opened_on_day = True

                        # Calculate risk reward, filter risk rewards that are not at least 1:1.5
                        if self.data.Close[-1] < self.previous_day_vah:
                            if (self.data.Close[-1] - self.data.Low[-1]) * 1.5 < (self.previous_day_vah - self.data.Close[-1]):
                                self.buy(size=0.01, sl=self.data.Low[-1].item(), tp=self.previous_day_vah)

                elif self.close_above_vah_day and not self.position:
                    if self.data.Close[-1] <= self.previous_day_vah and self.position_opened_on_day is False:
                        self.position_opened_on_day = True

                        if self.data.Close[-1] > self.previous_day_val:
                            if (self.data.High[-1] - self.data.Close[-1]) * 1.5 < (self.data.Close[-1] - self.previous_day_val):
                                self.sell(size=0.01, sl=self.data.High[-1].item(), tp=self.previous_day_val)

        except Exception as e:
            print(f"Error in next method: {e} at {self.data.index[-1]}")
            sys.exit()



try:
    df = tools.extract_candles_csv('BTC_2020-2024_1h.csv', datetime(2023, 1, 1), datetime(2024, 6, 28))
    # df = tools.extract_candles_csv('BTC_2024_1h.csv', datetime(2024, 7, 1), datetime(2024, 7, 23))
    print(df)

    factor = 100
    bt = Backtest(df, SessionVolumeStrategy, cash=1000000, margin=0.01, trade_on_close=True)

    ## Backtest Execution
    stats = bt.run()
    print(stats)
    bt.plot(plot_volume=True, superimpose=False, open_browser=True, resample=False, plot_drawdown=True)

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
