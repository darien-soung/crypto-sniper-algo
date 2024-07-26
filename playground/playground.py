import tools
from datetime import datetime, timedelta, time
import statsmodels.tsa.stattools as ts
import os
import pandas as pd
from private_indicators import is_price_away, is_price_within, session_volume_profile, SessionVolumeProfile
import numpy as np

df = tools.extract_candles_binance(datetime(2024, 7, 1), datetime(2024, 7, 24), "1h", 'BTCUSDT')
tools.df_to_csv(df, "BTC_2024_1h.csv")
print(df)

extracted_df = tools.extract_candles_csv(r'BTC_2024_1h.csv', datetime(2024, 7, 1), datetime(2024, 7, 24))
print(extracted_df)

tools.check_dataset_interval(timedelta(hours=1), 'BTC_2024_1h.csv')

# df = tools.extract_candles_csv(r'BTC_2020-2024_1h.csv', datetime(2020, 1, 23), datetime(2020, 1, 25))
# print(df)
# vah, poc, val = session_volume_profile(data=df, lookback=24)
# df["VAH"] = vah
# df["POC"] = poc
# df["VAL"] = val
# print("With the SVP indicators now")
# print(df.to_string())
