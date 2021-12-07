import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Gather hard-coded variables
# TODO CapsLock to the variables
flt_market_cap_limit = 10E11
str_default_ticker = "GME"
str_default_period = "5y"
str_interval = "1d"
str_csv_format = "_historical_data.csv"


def get_ticker_historical_data(ticker_code,
                               data_period=str_default_period,
                               use_persisted_data=True,
                               persist_data=False):
    """
    Function fetches daily historical price data for the given ticker.
    The datasource is either live Yahoo Finance API or data persisted in csv.
    :param ticker_code: str
        Equity asset ticker code.
    :param data_period: str, optional
        Period of data to fetch from Yahoo Finance. Default is 5 years.
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max.
    :param use_persisted_data: bool, optional
        Load data from (ticker_code)_historical_data.csv file. Default is True.
    :param persist_data: bool, optional
        Persist fetched data to (ticker_code)_historical_data.csv file. Default is False.
    :return: DataFrame
        Historical data indexed by Date with columns Open, High, Low, Close, Volume, Dividends, Stock, Splits.
    """
    if use_persisted_data & persist_data:
        raise Exception("Can't load and stream data at the same time.")

    if use_persisted_data:
        historical_data = pd.read_csv(ticker_code + str_csv_format)
    else:
        ticker = yf.Ticker(ticker_code)
        market_cap = ticker.info['marketCap']
        if market_cap > flt_market_cap_limit:
            raise Exception(f"""This analysis is intended for assets with market cap up to {flt_market_cap_limit}. 
            Current asset market cap is {market_cap}, choose different ticker.""")
        historical_data = ticker.history(period=data_period, interval=str_interval, rounding=True)
        # TODO historical_data = historical_data[["Open", "High", "Low", "Close", "Volume"]]
        if persist_data:
            historical_data.to_csv(ticker_code + str_csv_format)
    return historical_data

# TODO change default argument values to unmutable type (tuple?)
def add_statistics(historical_data,
                   use_sign=True,
                   use_move=True,
                   use_range=True,
                   sma_windows=[15, 30, 100, 200],
                   ema_alphas=[0.5, 0.25],
                   momentum_windows=[5, 15, 30]):
    # Extract only relevant columns and store in temporary data frame
    # TODO remove redundant df -> df = historical_data.copy() <- always use a copy to avoid warnings
    df = historical_data[["Open", "High", "Low", "Close", "Volume"]].copy()

    #TODO clear up returns
    # Add daily price returns, future return and sign
    df["Return"] = df["Close"].pct_change(1)
    if use_sign:
        df["Sign"] = np.log(df["Return"] + 1)
    df["Return_future"] = df["Return"].shift(-1)

    # Add daily price move and range
    if use_move:
        df["Move"] = df["Close"] - df["Open"]
    if use_range:
        df["Range"] = df["High"] - df["Low"]

    # Add momentum statistics
    for window in momentum_windows:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()

    # Add simple moving average statistics
    for window in sma_windows:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()

    # Add exponential moving average statistics
    # adjust parameter = False to conform with requirements as per pandas.emw specification
    for alpha in ema_alphas:
        df[f"EMA_{alpha}"] = df["Close"].ewm(alpha=alpha, adjust=False).mean()

    return df.dropna()


def categorize_return_data(statistics_data, low_cutoff, high_cutoff):
    df = statistics_data[(statistics_data["Return_future"] >= high_cutoff) | (statistics_data["Return_future"] <= low_cutoff)].copy()
    df["Return_direction"] = np.where(df["Return_future"] > 0, 1, -1)
    return df["Return_direction"]


if __name__ == '__main__':
    # test the statistics functions and visualize them on plots

    # load and save data
    # data = get_ticker_historical_data(str_default_ticker, use_persisted_data=False, persist_data=True)
    # print(data.head())

    data = get_ticker_historical_data(str_default_ticker, use_persisted_data=True)
    data_with_statistics = add_statistics(data)
    data_with_statistics.plot(x="Date", y=["Return"])
    plt.show()
