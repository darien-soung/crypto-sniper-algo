import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def extract_data_binance(start_time, end_time, interval, symbol):  # Extract data
    df = pd.DataFrame()
    data = []

    # User Settings Panel
    api_library = '/fapi/v1/klines'  # https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data

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
        elif interval == '1m':
            start_time = start_time + timedelta(minutes=1500)

        elif interval == '1h':
            start_time = start_time + timedelta(hours=1500)


    # Arrange and manage response data
    df = pd.DataFrame(data)

    combined_rows = []
    for _, row in df.iterrows():
        combined_row = []
        for cell in row:
            combined_row.extend(cell if cell is not None else [np.nan, np.nan, np.nan])
        combined_rows.append(combined_row)
    #print(combined_rows[0])

    #Split data into rows of 12 elements
    split_rows = [row[i:i + 12] for row in combined_rows for i in range(0, len(row), 12)]

    #Change the Open and Close time into Pandas.Datetime format from POSIX time format as given by Binance
    split_df = pd.DataFrame(split_rows)
    split_df[0] = pd.to_datetime(split_df[0], unit="ms")
    split_df[6] = pd.to_datetime(split_df[6], unit="ms")

    # Rename the columns for better understanding in csv
    split_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote Asset Volume',
                        'Number of Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    final_df = split_df.set_index("Open Time")
    # print(final_df)

    # Make the CSV
    #csv_df = final_df.to_csv('premium_index_transposed.csv')
    return final_df


extracted_data = extract_data_binance(datetime(2020, 5, 22), datetime(2024, 5, 22), "1d", 'BTCUSDT')
backtest_table = extracted_data[['Close']].copy()


# Create Moving Average column
ma_length = 112   # Length of moving average
backtest_table['Moving_Average'] = backtest_table['Close'].rolling(window=ma_length).mean()
# Parse the values into computable datatypes
backtest_table['Close'] = pd.to_numeric(backtest_table['Close'], errors='coerce')
backtest_table['Moving_Average'] = pd.to_numeric(backtest_table['Moving_Average'], errors='coerce')

# Create Signal to enter position: 1 = LONG, -1 = SHORT, and price difference for reference
threshold = 0.01 # Threshold percentage is 1%

def calculate_signal(close_price, moving_average):
    try:
        return 1 if close_price > moving_average and difference_in_percentage(close_price, moving_average) > threshold else -1 if close_price < moving_average and difference_in_percentage(close_price, moving_average) < np.negative(threshold) else 0
    except ValueError:
        return 'NaN'

def difference_in_percentage(x, y):
    return (x-y)/y


backtest_table['Difference'] = backtest_table.apply(lambda row: difference_in_percentage(row['Close'], row['Moving_Average']), axis=1)
backtest_table['Signal'] = backtest_table.apply(lambda row: calculate_signal(row['Close'], row['Moving_Average']), axis=1)


# Add Daily PnL and Cumulative PnL
def calculate_pnl(today_price, future_price, signal):
    return future_price / today_price - 1 if signal == 1 else np.negative(future_price / today_price - 1) if signal == -1 else 0


backtest_table['Future_Close'] = backtest_table['Close'].shift(-1)
backtest_table['PnL'] = backtest_table.apply(lambda row: calculate_pnl(row['Close'], row['Future_Close'], row['Signal']), axis=1)
backtest_table['Cumulative_PnL'] = backtest_table['PnL'].cumsum()

# Drop previous close column after processing is done.
backtest_table = backtest_table.drop(columns=['Future_Close'])
sharpe_ratio = backtest_table['PnL'].mean()/backtest_table['PnL'].std()*np.sqrt(365)

print(backtest_table)
print("Sharpe Ratio: ", sharpe_ratio)
csv_df = backtest_table.to_csv('daily_ma_strategy.csv')
