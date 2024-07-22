import numpy as np
import pandas as pd
from typing import List, Union
import tools


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

    for i in range(period, len(low) - period):
        if low[i] == min(low[i - period:i + period + 1]):
            pivot_lows[i] = low[i]

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


def session_volume_profile(data):
    svp = SessionVolumeProfile()
    highs = data['High'].values
    lows = data['Low'].values
    volumes = data['Volume'].values

    return svp.calculate(highs, lows, volumes)


### Indicator Classes go here ###

class SessionVolumeProfile:
    def __init__(self, number_of_rows=24, value_area_coverage=70, track_developing_va=False):
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
            price_portion = row_range
        elif candle_high <= row_high and candle_low >= row_low:
            price_portion = candle_range
        elif candle_high > row_high and candle_low >= row_low:
            price_portion = row_high - candle_low
        elif candle_high <= row_high and candle_low < row_low:
            price_portion = candle_high - row_low
        else:
            price_portion = 0

        return (price_portion * candle_volume) / candle_range

    def calculate(self, highs, lows, volumes):
        """
        This method performs the main calculation of the volume profile by iterating through each candle and each row,
        calculating the volume for each row, and then determining the Point of Control (POC), Value Area High (VAH),
        and Value Area Low (VAL).
        """
        day_high = np.max(highs)
        day_low = np.min(lows)
        step = (day_high - day_low) / self.number_of_rows

        volume_rows = np.zeros(self.number_of_rows)
        price_level_rows = np.linspace(day_low, day_high, self.number_of_rows)

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


