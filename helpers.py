import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Gather hard-coded variables
FLT_MARKET_CAP = 1E11
STR_DEFAULT_TICKER = "GME"
STR_DEFAULT_PERIOD = "5y"
STR_INTERVAL = "1d"
STR_CSV_FORMAT = "_historical_data.csv"
LST_MANDATORY_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def get_ticker_historical_data(ticker_code,
                               data_period=STR_DEFAULT_PERIOD,
                               use_persisted_data=False,
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
        Load data from (ticker_code)_historical_data.csv file. Default is False.
    :param persist_data: bool, optional
        Persist fetched data to (ticker_code)_historical_data.csv file. Default is False.
    :return: DataFrame
        Historical data indexed by Date with columns Open, High, Low, Close, Volume, Dividends, Stock, Splits.
    """

    if use_persisted_data and persist_data:
        raise Exception("Can't load and stream data at the same time.")

    if use_persisted_data:
        historical_data = pd.read_csv(ticker_code + STR_CSV_FORMAT)
    else:
        ticker = yf.Ticker(ticker_code)
        market_cap = ticker.info['marketCap']
        if market_cap > FLT_MARKET_CAP:
            raise Exception(f"""This analysis is intended for assets with market cap up to {FLT_MARKET_CAP}. 
            Current asset market cap is {market_cap}, choose different ticker.""")
        historical_data = ticker.history(period=data_period, interval=STR_INTERVAL, rounding=True)
        if persist_data:
            historical_data.to_csv(ticker_code + STR_CSV_FORMAT)
    return historical_data


def add_statistics(historical_data,
                   use_sign=False,
                   use_move=False,
                   use_range=False,
                   sma_windows=(),
                   ema_alphas=(),
                   momentum_windows=()):
    """
    Function calculates specified time series statistics.
    Features Return and Return_future are always present in the output.
    Possible
    :param historical_data: DataFrame
        Asset daily price data containing columns Open, High, Low, Close, Volume.
    :param use_sign: bool, optional
        Calculate sign feature. Default is False.
    :param use_move: bool, optional
        Calculate daily price move feature. Default is False.
    :param use_range: bool, optional
        Calculate daily price range. Default is False.
    :param sma_windows: tuple, optional
        Lookback periods for calculating simple moving average. Default is empty tuple.
    :param ema_alphas: tuple, optional
        Alpha parameters for calculating exponential moving average. Default is empty tuple.
    :param momentum_windows: tuple, optional
        Periods for calculating momentum. Default is empty tuple.
    :return: DataFrame
        Data contains historical data and calculated statistics.
    """

    # Extract only relevant columns and store in temporary data frame
    for column in LST_MANDATORY_COLUMNS:
        if not (column in historical_data):
            raise Exception(f"""Historical data not complete, {column} is missing.""")
    df = historical_data[["Open", "High", "Low", "Close", "Volume"]].copy()

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


def prepare_regression_data(data, feature_names, low_cutoff=0.0, high_cutoff=0.0):
    """
    Function prepares statistics data for the use in regression model. Following steps are done:
    1. Remove data with Return in cutoff interval.
    2. Select only desired features.py
    3. Scale data to normal distributions
    4. Categorize Return data
    5. Split data to features and signal; train and test data.
    :param data: DataFrame
        Data with calculated statistics.
    :param feature_names: list string
        Features to use as explanatory variables.
    :param low_cutoff: float, optional
        Lower bound of returns cutoff.
    :param high_cutoff: float, optional
        Upper bound for returns cutoff.
    :return: tuple DataFrame
        X_train, X_test, y_train, y_test
    """

    # remove data with returns within cutoff
    data_filtered = data[(data["Return_future"] >= high_cutoff) | (data["Return_future"] <= low_cutoff)].copy()

    # select only desired features
    x = data_filtered[feature_names]

    # scale the features data to have normal distributions
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # categorize data
    y_categorized = np.where(data_filtered["Return_future"] > 0, 1, -1)

    # split data randomly (but with set seed) to train and test population
    return train_test_split(x_scaled, y_categorized, test_size=0.25, random_state=1)
