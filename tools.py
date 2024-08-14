import sys

import pandas
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import statsmodels.tsa.stattools as ts
import os


def extract_candles_binance(start_time: datetime, end_time: datetime, interval, symbol):  # Extract data
    df = pd.DataFrame()
    data = []

    # User Settings Panel
    api_library = '/fapi/v1/klines'  # https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data

    if not isinstance(start_time, datetime) and not isinstance(end_time, datetime):
        raise ValueError(f"Start or End time provided is invalid")

    # Pull data from Binance REST API
    while start_time < end_time:
        print(start_time)
        start_time_timestamp = int(start_time.timestamp() * 1000)  # convert into posix time
        url = 'https://fapi.binance.com' + str(api_library) + '?symbol=' + str(symbol) + '&interval=' + str(
            interval) + '&limit=1500&startTime=' + str(start_time_timestamp)
        response = requests.get(url)
        response = json.loads(response.content.decode())
        data.append(response)

        if interval == "1d":
            start_time = start_time + timedelta(days=1500) # Change te days= to minute based on the interval. e.g. 1m interval ; minutes=1500.
                                                        # This happens because the limit is set at 1500 at the api, so 1500 records will be added regardless based
                                                          # on the interval. Adjust as needed, e.g. interval = 5m, so minutes=1500*5 because 1500 records of 5m is given
        elif interval == '4h':
            start_time = start_time + timedelta(days=250)

        elif interval == '1h':
            start_time = start_time + timedelta(hours=1500)

        elif interval == '15m':
            start_time = start_time + timedelta(hours=375) # 1500 / 4 because 1h/15m = 4

        elif interval == '30m':
            start_time = start_time + timedelta(hours=750)

        elif interval == '5m':
            start_time = start_time + timedelta(hours=125)

        elif interval == '1m':
            start_time = start_time + timedelta(minutes=1500)

        else:
            raise ValueError(f"Interval {interval} provided to tools is invalid")

    # Arrange and manage response data
    df = pd.DataFrame(data)

    combined_rows = []
    for _, row in df.iterrows():
        combined_row = []
        for cell in row:
            combined_row.extend(cell if cell is not None else [np.nan, np.nan, np.nan])
        combined_rows.append(combined_row)

    #Split data into rows of 12 elements
    split_rows = [row[i:i + 12] for row in combined_rows for i in range(0, len(row), 12)]

    # Change the Open and Close time into Pandas.Datetime format from POSIX time format as given by Binance
    split_df = pd.DataFrame(split_rows)
    split_df[0] = pd.to_datetime(split_df[0], unit="ms")
    split_df[6] = pd.to_datetime(split_df[6], unit="ms")

    # Rename the columns for better understanding in csv
    split_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume',
                        'Number of Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

    split_df['Open'] = pd.to_numeric(split_df['Open'], errors='coerce')
    split_df['High'] = pd.to_numeric(split_df['High'], errors='coerce')
    split_df['Low'] = pd.to_numeric(split_df['Low'], errors='coerce')
    split_df['Close'] = pd.to_numeric(split_df['Close'], errors='coerce')
    split_df['Volume'] = pd.to_numeric(split_df['Volume'], errors='coerce')

    final_df = split_df[split_df['Open Time'] <= end_time] # remove extra dates if exists

    final_df = final_df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    final_df = final_df.set_index("Open Time")

    # Make the CSV
    # csv_df = final_df.to_csv('test_extract.csv')
    return final_df

def extract_candles_csv(csv_filename, start_time: datetime, end_time: datetime) -> pandas.DataFrame:
    """

    :param csv_filename: r'BTC_2020-2024_5m.csv' Only accepts OHLCV format e.g. ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    :param start_time: datetime
    :param end_time: datetime
    :return: pandas Dataframe
    """

    folder_path = r"C:\Users\User\PycharmProjects\crypto-sniper-algo\trade_data"
    try:
        df = pd.read_csv(os.path.join(folder_path, csv_filename))
        print(f"Successfully imported csv data from {str(start_time)} to {str(end_time)}")
        df['Open Time'] = pd.to_datetime(df['Open Time'])
        filtered_df = df[(df['Open Time'] >= start_time) & (df['Open Time'] <= end_time)]
        filtered_df = filtered_df.set_index("Open Time")

        return filtered_df

    except Exception as e:
        print(f"Error in extracting from csv: {e}")
        sys.exit()


def df_to_csv(df, file_name):
    folder_path = r"C:\Users\User\PycharmProjects\crypto-sniper-algo\trade_data"
    df.to_csv(os.path.join(folder_path, file_name))


def extract_premium_index_binance(start_time: datetime, end_time: datetime, interval, symbol):
    df = pd.DataFrame()
    data = []

    # User Settings Panel
    api_library = '/fapi/v1/premiumIndexKlines'  # https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data

    if not isinstance(start_time, datetime) and not isinstance(end_time, datetime):
        raise ValueError(f"Start or End time provided is invalid")

    # Pull data from Binance REST API
    while start_time < end_time:
        print(start_time)
        start_time_timestamp = int(start_time.timestamp() * 1000)  # convert into posix time
        url = 'https://fapi.binance.com' + str(api_library) + '?symbol=' + str(symbol) + '&interval=' + str(
            interval) + '&limit=1500&startTime=' + str(start_time_timestamp)
        response = requests.get(url)
        response = json.loads(response.content.decode())
        data.append(response)

        if interval == "1d":
            start_time = start_time + timedelta(
                days=1500)  # Change te days= to minute based on the interval. e.g. 1m interval ; minutes=1500.
            # This happens because the limit is set at 1500 at the api, so 1500 records will be added regardless based
            # on the interval. Adjust as needed, e.g. interval = 5m, so minutes=1500*5 because 1500 records of 5m is given
        elif interval == '4h':
            start_time = start_time + timedelta(days=250)

        elif interval == '1h':
            start_time = start_time + timedelta(hours=1500)

        elif interval == '15m':
            start_time = start_time + timedelta(hours=375)  # 1500 / 4 because 1h/15m = 4

        elif interval == '30m':
            start_time = start_time + timedelta(hours=750)

        elif interval == '5m':
            start_time = start_time + timedelta(hours=125)

        elif interval == '1m': # 1 minute
            start_time = start_time + timedelta(hours=25)

        elif interval == '1M': # 1 month
            start_time = start_time + timedelta(minutes=1500)

        else:
            raise ValueError(f"Interval {interval} provided to tools is invalid")

        # Arrange and manage response data
        df = pd.DataFrame(data)

        combined_rows = []
        for _, row in df.iterrows():
            combined_row = []
            for cell in row:
                combined_row.extend(cell if cell is not None else [np.nan, np.nan, np.nan])
            combined_rows.append(combined_row)

        # Split data into rows of 12 elements
        split_rows = [row[i:i + 12] for row in combined_rows for i in range(0, len(row), 12)]

        # Change the Open and Close time into Pandas.Datetime format from POSIX time format as given by Binance
        split_df = pd.DataFrame(split_rows)
        split_df[0] = pd.to_datetime(split_df[0], unit="ms")
        split_df[6] = pd.to_datetime(split_df[6], unit="ms")

        # Rename the columns for better understanding in csv
        split_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Ignore', 'Close time', 'Ignore',
                            'Ignore', 'Ignore', 'Ignore', 'Ignore']

        split_df['Open'] = pd.to_numeric(split_df['Open'], errors='coerce')
        split_df['High'] = pd.to_numeric(split_df['High'], errors='coerce')
        split_df['Low'] = pd.to_numeric(split_df['Low'], errors='coerce')
        split_df['Close'] = pd.to_numeric(split_df['Close'], errors='coerce')

        final_df = split_df[split_df['Open Time'] <= end_time]  # remove extra dates if exists

        final_df = final_df[['Open Time', 'Open', 'High', 'Low', 'Close']]
        final_df = final_df.set_index("Open Time")

        # Make the CSV
        # csv_df = final_df.to_csv('test_extract.csv')
        return final_df


def check_dataset_interval(interval: timedelta, csv_filename): # Checks if the dataset has evenly paced intervals
    # Interval must be in timedelta format
    folder_path = r"C:\Users\User\PycharmProjects\Cybotrade\trade_data"
    df = pd.read_csv(os.path.join(folder_path, csv_filename))
    df['Open Time'] = pd.to_datetime(df['Open Time'])

    df['time_diff'] = df['Open Time'].diff()
    are_intervals_equal = df['time_diff'].iloc[1:] == interval  # Skip the first NaT value

    if are_intervals_equal.all():
        print(f"All times are equally spaced in {interval} intervals.")
    else:
        print(f"Not all times are equally spaced in {interval} intervals.")


def find_percentage(close, tpsl):
    if close > tpsl: # Long
        return (close - tpsl) / close
    elif close < tpsl: # Short
        return (tpsl - close) / close
    else:
        return 0

def get_mean(array):
    total = 0
    for i in range(0, len(array)):
        total += array[i]
    return total / len(array)

def get_stddev(array):
    total = 0
    mean = get_mean(array)
    for i in range(0, len(array)):
        minus_mean = math.pow(array[i] - mean, 2)
        total += minus_mean

    return math.sqrt(total / (len(array) - 1))


def ad_fuller_test():
    df = extract_candles_binance(datetime(2023, 6, 22), datetime(2024, 6, 24), "5m", 'BTCUSDT')
    # df.to_csv('BTC_2023-2024.csv')
    print(ts.adfuller(df['Close'], 1))