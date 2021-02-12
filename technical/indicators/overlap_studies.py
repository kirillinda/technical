"""
Overlap studies
"""
from numba import jit
import numpy as np
from numpy.core.records import ndarray
from pandas import DataFrame, Series
import talib.abstract as ta

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

@jit(nopython=True)
def PMAX(pkey, masrc,high, low, close, period=10, multiplier=3, length=12, MAtype=1):
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
    atr = np.array([])
    mavalue = np.array([])
    atr = ta.ATR(high, low, close, timeperiod=period)
    pm = pkey
    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 6:
        mavalue = ta.WMA(close, timeperiod=length)
    # Compute basic upper and lower bands
    basic_ub = mavalue + (multiplier * atr)
    basic_lb = mavalue - (multiplier * atr)
    # Compute final upper and lower bands
    final_ub = np.array([])
    final_lb = np.array([])
    for i in range(period, len(close)):
        final_ub[i] = basic_ub[i] if basic_ub[i] < final_ub[i - 1] or \
                                                         mavalue[i - 1] > final_ub[i - 1] else \
            final_ub[i - 1]
        final_lb[i] = basic_lb[i] if basic_lb[i] > final_lb[i - 1] or \
                                                         mavalue[i - 1] < final_lb[i - 1] else \
            final_lb[i - 1]

    # Set the Pmax value
    pm = np.array([])
    for i in range(period, len(close)):
        pm[i] = final_ub[i] if pm[i - 1] == final_ub[i - 1] and mavalue[
            i] <= final_ub[i] else \
            final_lb[i] if pm[i - 1] == final_ub[i - 1] and mavalue[i] > \
                                     final_ub[i] else \
                final_lb[i] if pm[i - 1] == final_lb[i - 1] and mavalue[i] >= \
                                         final_lb[i] else \
                    final_ub[i] if pm[i - 1] == final_lb[i - 1] and mavalue[i] < \
                                             final_lb[i] else 0.00

#    df.fillna(0, inplace=True)

    return pm


def DATATABLE(pkey, period, MAtype, multiplier, length, data_dict, masrc, high, low, close):
    data_dict[pkey] = \
        PMAX(pkey, masrc,high, low, close, period=period, multiplier=multiplier, length=length, MAtype=MAtype)
