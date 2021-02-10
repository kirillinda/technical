"""
Overlap studies
"""

from numpy.core.records import ndarray
from pandas import DataFrame, Series



########################################
#
# Overlap Studies Functions
#

# BBANDS               Bollinger Bands
def bollinger_bands(dataframe: DataFrame, period: int = 21, stdv: int = 2,
                    field: str = 'close', colum_prefix: str = "bb") -> DataFrame:
    """
    Bollinger bands, using SMA.
    Modifies original dataframe and returns dataframe with the following 3 columns
        <column_prefix>_lower, <column_prefix>_middle, and <column_prefix>_upper,
    """
    rolling_mean = dataframe[field].rolling(window=period).mean()
    rolling_std = dataframe[field].rolling(window=period).std()
    dataframe[f"{colum_prefix}_lower"] = rolling_mean - (rolling_std * stdv)
    dataframe[f"{colum_prefix}_middle"] = rolling_mean
    dataframe[f"{colum_prefix}_upper"] = rolling_mean + (rolling_std * stdv)

    return dataframe


# DEMA                 Double Exponential Moving Average

# EMA                  Exponential Moving Average
def ema(dataframe: DataFrame, period: int, field='close') -> Series:
    """
    Wrapper around talib ema (using the abstract interface)
    """
    import talib.abstract as ta
    return ta.EMA(dataframe, timeperiod=period, price=field)


# HT_TRENDLINE         Hilbert Transform - Instantaneous Trendline
# KAMA                 Kaufman Adaptive Moving Average
# MA                   Moving average
# MAMA                 MESA Adaptive Moving Average
# MAVP                 Moving average with variable period
# MIDPOINT             MidPoint over period
# MIDPRICE             Midpoint Price over period
# SAR                  Parabolic SAR
# SAREXT               Parabolic SAR - Extended

# SMA                  Simple Moving Average
def sma(dataframe, period, field='close'):
    import talib.abstract as ta
    return ta.SMA(dataframe, timeperiod=period, price=field)


# T3                   Triple Exponential Moving Average (T3)

# TEMA                 Triple Exponential Moving Average
def tema(dataframe, period, field='close'):
    import talib.abstract as ta
    return ta.TEMA(dataframe, timeperiod=period, price=field)


# TRIMA                Triangular Moving Average
# WMA                  Weighted Moving Average

# Other Overlap Studies Functions
def hull_moving_average(dataframe, period, field='close') -> ndarray:
    from pyti.hull_moving_average import hull_moving_average as hma
    return hma(dataframe[field], period)


def vwma(df, window):
    return (df['close'] * df['volume']).rolling(window).sum() / df.volume.rolling(window).sum()


def zema(dataframe, period, field='close'):
    """
    zero lag ema
    :param dataframe:
    :param period:
    :param field:
    :return:
    """
    dataframe = dataframe.copy()
    dataframe['ema1'] = ema(dataframe, period, field)
    dataframe['ema2'] = ema(dataframe, period, 'ema1')
    dataframe['d'] = dataframe['ema1'] - dataframe['ema2']
    dataframe['zema'] = dataframe['ema1'] + dataframe['d']
    return dataframe['zema']


def PMAX(dataframe, period=10, multiplier=3, length=12, MAtype=1, src=1):
    """
    Function to compute PMAX

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        length: moving averages length
        MAtype: type of the moving average

    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR), ATR (ATR_$period)
            PMAX (pm_$period_$multiplier_$length_$Matypeint)
            PMAX Direction (pmX_$period_$multiplier_$length_$Matypeint)
    """
    import numpy as np
    import talib.abstract as ta
    df = dataframe.copy()
    mavalue = 'MA_' + str(MAtype) + '_' + str(length)
    atr = 'ATR_' + str(period)
    df[atr] = ta.ATR(df, timeperiod=period)
    pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4
    if MAtype == 1:
        df[mavalue] = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        df[mavalue] = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        df[mavalue] = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        df[mavalue] = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        df[mavalue] = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 6:
        df[mavalue] = ta.WMA(df, timeperiod=length)
    elif MAtype == 7:
        df[mavalue] = vwma(df, length)
    elif MAtype == 8:
        df[mavalue] = zema(df, period=length)
    # Compute basic upper and lower bands
    df['basic_ub'] = df[mavalue] + (multiplier * df[atr])
    df['basic_lb'] = df[mavalue] - (multiplier * df[atr])
    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df[mavalue].iat[i - 1] > df['final_ub'].iat[i - 1] else \
        df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df[mavalue].iat[i - 1] < df['final_lb'].iat[i - 1] else \
        df['final_lb'].iat[i - 1]

    # Set the Pmax value
    df[pm] = 0.00
    for i in range(period, len(df)):
        df[pm].iat[i] = df['final_ub'].iat[i] if df[pm].iat[i - 1] == df['final_ub'].iat[i - 1] and df[mavalue].iat[
            i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[pm].iat[i - 1] == df['final_ub'].iat[i - 1] and df[mavalue].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[pm].iat[i - 1] == df['final_lb'].iat[i - 1] and df[mavalue].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[pm].iat[i - 1] == df['final_lb'].iat[i - 1] and df[mavalue].iat[i] < \
                                             df['final_lb'].iat[i] else 0.00
        # Mark the trend direction up/down
    df[pmx] = np.where((df[pm] > 0.00), np.where((df[mavalue] < df[pm]), 'down', 'up'), np.NaN)
    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df[pmx]
