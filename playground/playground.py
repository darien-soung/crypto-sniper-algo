import tools
from datetime import datetime, timedelta, time
import statsmodels.tsa.stattools as ts
import os
import pandas as pd
from private_indicators import is_price_away, is_price_within, session_volume_profile, SessionVolumeProfile
import numpy as np
from scipy.optimize import minimize

# df = tools.extract_candles_binance(datetime(2024, 7, 1), datetime(2024, 7, 24), "1h", 'BTCUSDT')
# tools.df_to_csv(df, "BTC_2024_1h.csv")
# print(df)
#
extracted_df = tools.extract_candles_csv(r'BTC_2020-2024_1h.csv', datetime(2020, 2, 1), datetime(2024, 7, 24))
# print(extracted_df)
#
# tools.check_dataset_interval(timedelta(hours=1), 'BTC_2024_1h.csv')

J = 5

extracted_df['returns'] = np.log(extracted_df['Close'] / extracted_df['Close'].shift(1))
extracted_df['r_squared'] = extracted_df['returns']**2
extracted_df['hl_range'] = np.log(extracted_df['High'] / extracted_df['Low'])
extracted_df['hl_range_squared'] = extracted_df['hl_range']**2
extracted_df['realized_vol_squared'] = extracted_df['returns'].rolling(window=J).apply(lambda x: np.sum(x**2), raw=True)
extracted_df.dropna(inplace=True)
print(extracted_df)


# Define MEM log-likelihood function
def mem_log_likelihood(params, data):
    """ Hourly data
    :param params:   alpha_r: 0.0004156745140832911, beta_r: 0.7997740801270903
                     alpha_hl: 0.09954843336627922, beta_hl: 0.799773856551866
                     alpha_v: 0.09954884452570464, beta_v: 0.7997738897490988
    :param data: the df
    :return:
    """
    alpha_r, beta_r, alpha_hl, beta_hl, alpha_v, beta_v = params

    h_r = np.var(data['r_squared']) * np.ones(len(data))
    h_hl = np.var(data['hl_range_squared']) * np.ones(len(data))
    h_v = np.var(data['realized_vol_squared']) * np.ones(len(data))

    for t in range(1, len(data)):
        h_r[t] = alpha_r + beta_r * h_r[t - 1] + data['r_squared'][t - 1]
        h_hl[t] = alpha_hl + beta_hl * h_hl[t - 1] + data['hl_range_squared'][t - 1]
        h_v[t] = alpha_v + beta_v * h_v[t - 1] + data['realized_vol_squared'][t - 1]

    ll_r = -np.log(h_r) - data['r_squared'] / h_r
    ll_hl = -np.log(h_hl) - data['hl_range_squared'] / h_hl
    ll_v = -np.log(h_v) - data['realized_vol_squared'] / h_v

    log_likelihood = np.sum(ll_r + ll_hl + ll_v)
    print(f"Log Likelihood: {-log_likelihood}")
    return -log_likelihood


initial_params = np.array([0.006909537384378606, 0.8, 0.1, 0.8, 0.1, 0.8])
result = minimize(mem_log_likelihood, initial_params, args=(extracted_df,), method='L-BFGS-B')
alpha_r_est, beta_r_est, alpha_hl_est, beta_hl_est, alpha_v_est, beta_v_est = result.x
print(f"Estimated Parameters:\n alpha_r: {alpha_r_est}, beta_r: {beta_r_est}\n alpha_hl: {alpha_hl_est}, beta_hl: {beta_hl_est}\n alpha_v: {alpha_v_est}, beta_v: {beta_v_est}")