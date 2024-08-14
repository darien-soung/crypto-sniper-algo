import numpy as np
import pandas as pd
from typing import List, Union
import tools
from datetime import datetime, time


def crossover(series1: Union[List[float], np.ndarray], series2: Union[List[float], float, int, np.ndarray]) -> int:
    """
    Determine crossover between two series.

    Parameters:
    - series1: List or array of floats representing the first series.
    - series2: List, float, int, or array representing the second series.

    Returns:
    - 1 if crossover occurs to the upside
    - -1 if crossover occurs to the downside
    - 0 otherwise.
    """
    # Convert series1 to a list if it is a ndarray
    if isinstance(series1, np.ndarray):
        series1 = series1.tolist()

    # Convert series2 to a list if it is a float or an int
    if isinstance(series2, (float, int)):
        series2 = [series2] * len(series1)  # Create a list of the same length as series1 with the same value
    elif isinstance(series2, np.ndarray):
        series2 = series2.tolist()

    # Ensure series2 is a list now
    if not isinstance(series2, list):
        raise ValueError("series2 must be a list, float, or int")

    if series1[-1] >= series2[-1] and series1[-2] < series2[-2]:
        return 1
    elif series1[-1] <= series2[-1] and series1[-2] > series2[-2]:
        return -1
    else:
        return 0

def crossover_zero(series1, lookback):
    for i in range(np.negative(lookback)+1, 0):
        if series1[i] > 0 >= series1[i-1]:
            return 1
        elif series1[i] < 0 <= series1[i-1]:
            return -1
        else:
            return 0

def zscore(close_array, window):
    """
    Compute the z-score of the close_array over a rolling window for all rows.

    Parameters:
    - close_array (np.ndarray): Array of closing prices.
    - window (int): Rolling window size for calculating mean and standard deviation.

    Returns:
    - np.ndarray: Array of z-score values corresponding to each row in close_array.
    """
    z_scores = np.zeros_like(close_array)  # Initialize array for z-scores

    for i in range(window - 1, len(close_array)):
        # Calculate rolling mean and standard deviation
        rolling_mean = np.mean(close_array[i - window + 1: i + 1])
        rolling_std = np.std(close_array[i - window + 1: i + 1])

        # Calculate z-score
        z_scores[i] = (close_array[i] - rolling_mean) / rolling_std

    return z_scores


def pivot_high(high, period=5):
    """
    Identify pivot high points.

    :param high: Series of high prices
    :param period: Number of periods before and after the pivot point to compare
    :return: A Series with pivot high points (value of high at pivot high, NaN otherwise)
    """
    pivot_highs = np.full_like(high, np.nan)

    for i in range(period, len(high) - period):
        if high[i] == max(high[i - period:i + period + 1]):
            pivot_highs[i] = high[i]

    return pivot_highs


def pivot_low(low, period=5):
    """
    Identify pivot low points.

    :param low: Series of low prices
    :param period: Number of periods before and after the pivot point to compare
    :return: A Series with pivot low points (value of low at pivot low, NaN otherwise)
    """
    pivot_lows = np.full_like(low, np.nan)
    pivot_backcandles_index = 1 + (period * 2)
    for i in range(pivot_backcandles_index, len(low)):
        middle_index = i-period+1
        if low[middle_index] == min(low[i - pivot_backcandles_index:i]): # checks if the middle index is a pivot low/high
            pivot_lows[middle_index] = low[middle_index]

    return pivot_lows


def is_price_away(price_a, price_b, percentage=0.5):
    # Calculate the absolute difference
    difference = abs(price_a - price_b)

    # Calculate the threshold (0.5% of price B)
    threshold = (percentage / 100) * price_b

    # Check if the difference is at least the threshold
    return difference >= threshold


def is_price_within(price_a, price_b, percentage=0.5):
    # Calculate the absolute difference
    difference = abs(price_a - price_b)

    # Calculate the threshold (0.5% of price B)
    threshold = (percentage / 100) * price_b

    # Check if the difference is within the threshold
    return difference <= threshold


def calculate_realized_volatility(high, low, close, rolling_window):
    returns = np.full_like(close, np.nan)

    # Calculate logarithmic returns
    returns = np.log(close[1:] / close[:-1])
    realized_vol_squared = np.full_like(close, np.nan)

    for i in range(rolling_window - 1, len(returns)):
        window_returns = returns[i - rolling_window + 1:i + 1]
        realized_vol_squared[i] = np.sum(window_returns ** 2)

    return realized_vol_squared


def calculate_bollinger_volatility(upper_band, lower_band, window, sma):
    # Convert inputs to pandas Series if they are not already
    upper_band = pd.Series(upper_band)
    lower_band = pd.Series(lower_band)
    sma = pd.Series(sma)

    # Calculate Bollinger Band width
    width = upper_band - lower_band

    # Normalize the Bollinger Band width to measure relative volatility
    normalized_width = width / sma

    # Initialize lists to store thresholds and volatility levels
    low_volatility_thresholds = []
    high_volatility_thresholds = []
    volatility_level = []

    # Loop through each value to ensure thresholds are in ascending order
    for i in range(len(normalized_width)):
        current_data = normalized_width[:i + 1]  # Use data up to the current point

        low_volatility_threshold = current_data.quantile(0.15)
        high_volatility_threshold = current_data.quantile(0.85)

        # Ensure thresholds are ordered
        if i > 0:
            low_volatility_threshold = max(low_volatility_threshold, low_volatility_thresholds[-1])
            high_volatility_threshold = max(high_volatility_threshold, high_volatility_thresholds[-1])

        low_volatility_thresholds.append(low_volatility_threshold)
        high_volatility_thresholds.append(high_volatility_threshold)

        # Categorize the current normalized width
        if normalized_width[i] < low_volatility_threshold:
            volatility_level.append(0)  # Low volatility
        elif normalized_width[i] > high_volatility_threshold:
            volatility_level.append(2)  # High volatility
        else:
            volatility_level.append(1)  # Medium volatility

    return np.array(volatility_level)

def session_volume_profile(date, highs, lows, volumes, lookback):
    """
    SVP indicator
    :param data: must be OHLCV data with Open Time as index e.g. Hourly chart, lookback = 24. Data must start with 00:00 OR else it returns error
    :param lookback: based on how many candles form a day relative to the timeframe
    :return:
    """
    svp = SessionVolumeProfile()

    first_datetime = date[0]
    if first_datetime.time() == time(0, 0) and len(highs) == len(lows) == len(volumes):

        vah_list = np.full(len(highs), np.nan)
        poc_list = np.full(len(highs), np.nan)
        val_list = np.full(len(highs), np.nan)

        for i in range(0, len(highs), lookback):
            day_highs = highs[i: i + lookback].tolist()
            day_lows = lows[i: i + lookback].tolist()
            day_volumes = volumes[i: i + lookback].tolist()

            vah, poc, val = svp.calculate(day_highs, day_lows, day_volumes, lookback, date[i])

            vah_list[i:i+lookback] = vah
            poc_list[i:i+lookback] = poc
            val_list[i:i + lookback] = val

        return [vah_list, poc_list, val_list]

    else:
        raise Exception("Data frame time doesn't start at time 00:00 or length of highs lows and volumes are different")


### Indicator Classes go here ###

class SessionVolumeProfile:
    def __init__(self, number_of_rows=24, value_area_coverage=80, track_developing_va=False):
        self.number_of_rows = number_of_rows
        self.value_area_coverage = value_area_coverage / 100
        self.track_developing_va = track_developing_va

    def get_volume_for_row(self, row_high, row_low, candle_high, candle_low, candle_volume):
        """
        This method calculates the portion of the candle's volume that should be allocated to a specific row
        based on the overlap of the candle's price range with the row's price range.
        """
        row_range = row_high - row_low
        candle_range = candle_high - candle_low
        price_portion = 0

        if candle_high > row_high and candle_low < row_low:
            # The candle engulfed the row, so take the entire row range's portion of volume out of the candle
            price_portion = row_range
        elif candle_high <= row_high and candle_low >= row_low:
            # The top of the candle is in-or-equal the row, the bottom of the candle is also in-or-equal the row
            price_portion = candle_range
        elif candle_high > row_high and candle_low >= row_low and candle_low < row_high:
            # The top of the candle is above the row, and the bottom is in-or-equal the row
            price_portion = row_high - candle_low
        elif candle_high <= row_high and candle_low < row_low and candle_high > row_low:
            # The top of the candle is in-or-equal the row, the bottom of the candle is below the row
            price_portion = candle_high - row_low
        else:
            # The candle didn't intersect with the row at all
            price_portion = 0

            # return the portion of volume from the candle relative to the amount of price intersecting the row
        return (price_portion * candle_volume) / candle_range

    def calculate(self, highs, lows, volumes, number_of_rows, date):
        """
        This method performs the main calculation of the volume profile by iterating through each candle and each row,
        calculating the volume for each row, and then determining the Point of Control (POC), Value Area High (VAH),
        and Value Area Low (VAL).
        """
        self.number_of_rows = number_of_rows
        day_high = np.max(highs)
        day_low = float(np.min(lows))
        step = (day_high - day_low) / self.number_of_rows

        volume_rows = np.zeros(self.number_of_rows)
        price_level_rows = np.linspace(day_low, day_high, self.number_of_rows, endpoint=False)

        for i in range(len(highs)):
            for j in range(self.number_of_rows):
                row_high = price_level_rows[j] + step
                row_low = price_level_rows[j]
                volume_rows[j] += self.get_volume_for_row(row_high, row_low, highs[i], lows[i], volumes[i])

        poc_level = np.argmax(volume_rows)
        highest_vol = volume_rows[poc_level]

        value_area_tracking = volume_rows[poc_level]
        total_vol = np.sum(volume_rows)
        value_area_vol = total_vol * self.value_area_coverage

        vah_level = poc_level
        val_level = poc_level

        while value_area_tracking < value_area_vol:
            volume_above_poc = volume_rows[vah_level + 1] if vah_level + 1 < self.number_of_rows else 0
            volume_below_poc = volume_rows[val_level - 1] if val_level - 1 >= 0 else 0

            if volume_above_poc >= volume_below_poc:
                value_area_tracking += volume_above_poc
                vah_level += 1
            else:
                value_area_tracking += volume_below_poc
                val_level -= 1

        poc = price_level_rows[poc_level] + step / 2
        vah = price_level_rows[vah_level] + step
        val = price_level_rows[val_level]

        return vah, poc, val


