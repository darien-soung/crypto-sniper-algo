import numpy as np
import pandas as pd
from typing import List, Union
import tools
from datetime import datetime, time
import talib as ta

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


def stiffness_indicator(close_prices, ma_length_stiffness=100, stiff_length=60, stiff_smooth=3, threshold_stiffness=90):
    """ UNTESTED
    Calculate the Stiffness indicator values.

    Parameters:
    close_prices (array-like): Array of closing prices.
    ma_length_stiffness (int): Moving average length for stiffness.
    stiff_length (int): Length for stiffness calculation.
    stiff_smooth (int): Smoothing length for stiffness.
    threshold_stiffness (int): Threshold for stiffness.

    Returns:
    np.ndarray: Array of stiffness values.
    """

    close_prices = pd.Series(close_prices)

    # Calculate the moving average of close prices
    ma_close = close_prices.rolling(window=ma_length_stiffness).mean()

    # Calculate the standard deviation of close prices
    std_close = close_prices.rolling(window=ma_length_stiffness).std()

    # Calculate bound stiffness
    bound_stiffness = ma_close - 0.2 * std_close

    # Compare close prices to bound stiffness (True/False Series)
    above_bound = close_prices > bound_stiffness
    above_bound = pd.Series(above_bound)

    # Convert the boolean Series to an integer Series
    above_bound_int = above_bound.astype(int)

    # Calculate the sum above bound stiffness using a rolling window
    sum_above_stiffness = above_bound_int.rolling(window=stiff_length).sum()

    # Calculate the stiffness
    stiffness = sum_above_stiffness * 100 / stiff_length

    # Smooth the stiffness using EMA
    stiffness_smoothed = stiffness.ewm(span=stiff_smooth, adjust=False).mean()

    return stiffness_smoothed.values


def williams_percent_r(high, low, close, length=14, ema_length=24):
    """
    Calculate Williams %R values.

    Parameters:
    length (int): The lookback period for calculating Williams %R.

    Returns:
    pd.Series: A pandas series of Williams %R values.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()

    williams_r = 100 * (close - highest_high) / (highest_high - lowest_low)

    ema_wpr = calculate_ema(williams_r, ema_length)

    return williams_r, ema_wpr


def calculate_ema(series, length):
    # Calculate the Exponential Moving Average (EMA)
    series = pd.Series(series)
    ema = series.ewm(span=length, adjust=False).mean()
    return ema

def calculate_wma(series, period):
    """
    Calculate the Weighted Moving Average (WMA).

    Parameters:
    series (pd.Series): The data series (e.g., close prices).
    period (int): The period over which to calculate the WMA.

    Returns:
    pd.Series: WMA values.
    """
    weights = pd.Series(range(1, period + 1))
    wma = series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

def calculate_atr(high, low, close, atr_length):
    """
    Calculate the Average True Range (ATR).

    Parameters:
    high (array-like): Array of high prices.
    low (array-like): Array of low prices.
    close (array-like): Array of close prices.
    atr_length (int): The length for calculating the ATR.

    Returns:
    pd.Series: Array of ATR values.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/atr_length, adjust=False).mean() # Uses alpha instead of span, don't change

    return atr


def pine_atr(high, low, close, length=14):
    """
    Calculate the ATR using the Pine Script logic.
    length (int): Period for calculating ATR (default is 14).

    Returns:
    pd.Series: ATR values.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    # Calculate True Range (TR)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = np.where(prev_close.isna(), tr1, np.maximum(tr1, np.maximum(tr2, tr3)))

    # Calculate RMA (Rolling Moving Average, equivalent to Pine Script ta.rma)
    rma = pd.Series(true_range).ewm(span=length, adjust=False).mean()

    return rma


def coppock_curve(close, length=10, long_roc_length=14, short_roc_length=11, ma_length=10,
                  signal_logic='Zero line'): # options=['Zero line', 'Moving Average']
    """
    Calculate Coppock Curve and generate entry signals.

    Parameters:
    length (int): The smoothing length for Coppock Curve.
    long_roc_length (int): The long ROC length.
    short_roc_length (int): The short ROC length.
    ma_length (int): The length of the moving average for Coppock Curve.
    signal_logic (str): Signal logic, either 'Zero line' or 'Moving Average'.

    Returns:
    pd.DataFrame: A DataFrame with Coppock Curve, Coppock MA, and signals.
    """
    close = pd.Series(close)

    # Calculate ROC (Rate of Change)
    long_roc = close.pct_change(long_roc_length) * 100
    short_roc = close.pct_change(short_roc_length) * 100

    # Coppock Curve: WMA of the sum of long and short ROC
    # coppock = (long_roc + short_roc).rolling(window=length, min_periods=1).mean()
    coppock = calculate_wma(long_roc + short_roc, length)

    # Coppock MA: EMA of the Coppock Curve
    coppock_ma = calculate_ema(coppock, ma_length)

    # Signals
    entry_signal_long = (coppock > 0) if signal_logic == 'Zero line' else (coppock > coppock_ma)
    entry_signal_short = (coppock < 0) if signal_logic == 'Zero line' else (coppock < coppock_ma)

    # 0 line
    coppock_zero_line = np.zeros_like(coppock)

    return coppock, coppock_ma, coppock_zero_line

def trend_akkam(open, high, low, close, use_akkam=True, cross_akkam=False, inverse_akkam=False,
                     akk_range=50, ima_range=6, akk_factor=10.0, mode=0, delta_price=30.0):
    """
    AKKAM Trend Indicator
    Parameters:
    - open, high, low, close: Arrays of OHLC prices
    - use_akkam: Boolean, whether to use AKKAM logic
    - cross_akkam: Boolean, whether to use cross confirmation logic
    - inverse_akkam: Boolean, whether to invert signals
    - akk_range: ATR range
    - ima_range: MA range for EMA
    - akk_factor: ATR factor for DeltaStop calculation
    """

    # ATR Calculation
    atr_value = calculate_atr(high, low, close, akk_range) # atr_value = ta.ATR(high, low, close, timeperiod=akk_range)

    # DeltaStop Calculation
    delta_stop = calculate_ema(atr_value, ima_range) * akk_factor # delta_stop = ta.EMA(atr_value, timeperiod=ima_range) * akk_factor

    # Initialize TrStop with NaN
    tr_stop = np.full_like(open, np.nan)

    # Calculate TrStop
    for i in range(1, len(open)):
        if np.isnan(tr_stop[i - 1]):
            tr_stop[i] = open[i] - delta_stop[i]
        else:
            if open[i] == tr_stop[i - 1]:
                tr_stop[i] = tr_stop[i - 1]
            elif open[i - 1] < tr_stop[i - 1] and open[i] < tr_stop[i - 1]:
                tr_stop[i] = min(tr_stop[i - 1], open[i] + delta_stop[i])
            elif open[i - 1] > tr_stop[i - 1] and open[i] > tr_stop[i - 1]:
                tr_stop[i] = max(tr_stop[i - 1], open[i] - delta_stop[i])
            else:
                tr_stop[i] = open[i] - delta_stop[i] if open[i] > tr_stop[i - 1] else open[i] + delta_stop[i]

    # # Generate Signals
    # basic_long_condition = close > tr_stop
    # basic_short_condition = close < tr_stop
    #
    # long_signals = basic_long_condition
    # short_signals = basic_short_condition
    #
    # if cross_akkam:
    #     long_signals = np.logical_and(np.roll(long_signals, 1) == False, long_signals)
    #     short_signals = np.logical_and(np.roll(short_signals, 1) == False, short_signals)
    #
    # if inverse_akkam:
    #     long_signals, short_signals = short_signals, long_signals
    #
    # long_signals_final = long_signals if use_akkam else np.ones_like(long_signals) # can return these 2 if you want
    # short_signals_final = short_signals if use_akkam else np.ones_like(short_signals)

    return tr_stop


def atr_bands(high, low, close, atr_period=14, atr_multiplier_upper=2.0, atr_multiplier_lower=2.0):
    """
    Calculate the ATR bands.

    Parameters:
    atr_period (int): ATR period.
    atr_multiplier_upper (float): Multiplier for upper band.
    atr_multiplier_lower (float): Multiplier for lower band.
    """

    atr = pine_atr(high, low, close, atr_period)

    # Calculate the ATR bands
    lower_band = close - atr * atr_multiplier_lower
    upper_band = close + atr * atr_multiplier_upper

    return upper_band, lower_band

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


