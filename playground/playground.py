import tools
from datetime import datetime, timedelta
import statsmodels.tsa.stattools as ts
import os
from private_indicators import is_price_away, is_price_within

# df = tools.extract_candles_binance(datetime(2020, 1, 22), datetime(2024, 6, 24), "1d", 'BTCUSDT')
# tools.df_to_csv(df, "BTC_2020-2024_1d.csv")
# print(df)
#
# extracted_df = tools.extract_candles_csv(r'BTC_2020-2024_1d.csv', datetime(2020, 1, 22), datetime(2024, 6, 24))
# print(extracted_df)
#
# tools.check_dataset_interval(timedelta(days=1), 'BTC_2020-2024_1d.csv')\


close = 67900
ema_values = [80000, 67867, 67875, 67917, 68050]
print(all(ema_values[0] >= ema for ema in ema_values))



