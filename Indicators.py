import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def array_shift(arr:np.array,num:int)->np.array:
    """Shifts the arrray values and pads with nans as necessary."""
    new = np.empty_like(arr)
    if num >= 0:
        new[:num] = np.nan
        new[num:] = arr[:-num]
    else:
        new[num:] = np.nan
        new[:num] = arr[-num:]
    return new


def rescale(arr:np.array,low:int,high:int)->np.array:
    """Rescales the data to the desired range"""
    omax = np.nanmax(arr)
    omin = np.nanmin(arr)
    return (high-low)/(omax-omin)*(arr-omax)+high


def calculate_alpha(window:int,smoothing:int):
    """Calculates the alpha value for EMA. The prediction lag tells us how 
    much moving average will lag from the live value."""
    a = (smoothing / (1+window))
    return a


def exponential_moving_average(
        data:np.array,
        window:int,
        smoothing:int=2)->np.array:
    """Calculates the exponential moving average. Implemented from
    https://www.investopedia.com/terms/e/ema.asp
    
    Parameters:
        window (int): The number of lags to use in the moving average.
        smoothing (int): Used for alpha constant, determines smooth the ma
            will be."""
    alpha = calculate_alpha(window=window,smoothing=smoothing)
    
    current_idx = 0
    # Just like pandas we take the first window values as nans.
    window = window-1
    result = np.repeat(np.nan,window)

    while current_idx + window < len(data):
        # The first value of the average is just its the value*1.0
        alphas = np.array([1.0,*np.repeat(alpha,window)])
        
        # factors are alpha*(1-alpha)^n where n is Window -> 0
        factors = np.multiply(
            alphas[:-1],np.power(
                np.subtract(
                    np.ones(window),alphas[1:]),
                    np.arange(window-1,-1,-1)))
        # The result is the sum of factors*lag values.
        result = np.append(
            result,
            np.sum(
                np.multiply(
                    data[current_idx:(current_idx + window)],
                    factors)))
        current_idx += 1
    return result


def simple_moving_average(data:np.array,window:int):
    """Implementation of a simple moving average for numpy."""
    current_idx = 0
    window = window - 1
    result = np.repeat(np.nan,window)
    while current_idx + window < len(data):
        result = np.append(result,data[current_idx:current_idx+window].mean())
        current_idx += 1
    return result


def hull_moving_average(data:np.array,window:int):
    """Hull moving averages have a lower lag compared to other types. Using as
    3 stop exponential moving average. Implemented from:
    https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average
    """
    eMA1 = exponential_moving_average(data,window//2)
    eMA2 = exponential_moving_average(data,window)
    raw_hMA = (2*eMA1)-eMA2

    return exponential_moving_average(raw_hMA,int(np.sqrt(window)))


def kaufman_adaptive_moving_average(data:np.array,er_period:int,fast_ema:int,slow_ema:int)->np.array:
    """KAMA is a technique that adapts to the volatility of the market. If there
    are many swings then KAMA follows them closely, if there are few then it
    relaxes and follow from a distance. Implemented from
    https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
    """
    fast_sc = (2/(1+fast_ema))
    slow_sc = (2/(1+slow_ema))
    # The first KAMA value is just the simple moving average
    prior_kama = simple_moving_average(data[:er_period],er_period)[-1]

    index = 1
    final_kama = np.array([*np.repeat(np.nan,er_period-1),prior_kama])
    
    # Since for 10 periods ago we need the 11th period. we use a +1
    adj_er_period = er_period + 1

    while index + er_period - 1< len(data):
        temp = data[index:index+adj_er_period]
        change = np.abs(temp[-1] - data[index])
        volatility = np.sum(np.abs(np.diff(temp)))
        
        er = 0.0 if volatility==0.0 else change/volatility
        # Smoothing constant = Efficiency Ratio adjusted by the fast and slow ema.
        sc = np.power(er*(fast_sc-slow_sc)+slow_sc,2)

        kama = prior_kama + sc * (temp[-1] - prior_kama) 
        prior_kama = kama
        final_kama = np.append(final_kama,kama) 
        index += 1
    return final_kama


def on_balance_volume(data:np.array,period:int)->np.array:
    """Momentum indicator. Theory is that if volume increases without a big
    price change, a price change is coming. Implemented from:
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    
    Parameters:
        period (int): The size of period to use in calculation. Like kline 
            period"""
    shifted = array_shift(data,period)
    difference = np.subtract(data,shifted)
    
    mask = np.where(difference>0,1,difference)
    mask = np.where(difference<0,-1,mask)
    
    results = np.multiply(data,mask)
    results = np.nancumsum(results)

    return results


def bollinger_bands(data:np.array,window:int,factor:int=2):
    """Indicator used to identify whether a stock is overbought or oversold.
    Generally used in conjunciton with RSI and MACD. Use the bandwidth with 
    a long term moving average of itself to find squeezes. Implemented from:
    https://www.investopedia.com/terms/b/bollingerbands.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce

    Parameters:
        window (int): The lags to use for our moving average
        factor (int): The number of standard deviations to account for.
    Returns upper_band,moving_average,lower_band,bandwidth,%B"""
    upper_band = np.repeat(np.nan,window-1)
    lower_band = np.repeat(np.nan,window-1)
    simple_moving_average = np.repeat(np.nan,window-1)
    closes = np.repeat(np.nan,window-1)

    current_idx = 0
    while current_idx + window-1 < data.shape[0]:
        temp = data[current_idx:current_idx+window]
        simple_mean = temp.mean()
        std_dev = temp.std()*factor

        upper_band = np.append(upper_band,simple_mean+std_dev)
        lower_band = np.append(lower_band,simple_mean-std_dev)
        simple_moving_average = np.append(simple_moving_average,simple_mean)
        closes = np.append(closes,temp[-1])
        
        current_idx +=1

    percentB = (closes-lower_band)/(upper_band-lower_band)
    bandwidth = ((upper_band - lower_band) / simple_moving_average) * 100
    return upper_band,simple_moving_average,lower_band,bandwidth,percentB


def mean_average_convergance_divergance(
        data:np.array,fast_window:int,slow_window:int,signal_window:int):
    """Calculate the moving average convergance divergance. Momentum indicator
    which operates on the interaction between the MACD line and the Signal line.
    Implemented following: https://www.investopedia.com/terms/m/macd.asp
    
    Parameters:
        fast_window (int): The lags to use for the more reactive line
        slow_window (int): The lags to use for the more stable line. Should be
            more than for the fast window.
        signal_window (int): The window to use over the macd line to compare
            against. Should be less than both the others.

    Returns MACD,Signal,Histogram
    """
    fast = exponential_moving_average(data=data,window=fast_window)
    slow = exponential_moving_average(data=data,window=slow_window)
    macd_line = fast - slow
    signal = exponential_moving_average(data=macd_line,window=signal_window)
    histogram = macd_line - signal

    return macd_line,signal,histogram


def relative_strength_index(data:np.array,periods:int)->np.array:
    """Calculate Relative strength index for the data and periods specified.
    Evaluates overbought and oversold conditions. Values over 70 are overbought
    and values below 30 are oversold. Implemented following:
    https://www.investopedia.com/terms/r/rsi.asp"""
    # The first value in the calculation is a seed value, hence we start our
    # index at 1.
    price_changes = np.diff(data[0:periods+1])
    prev_gain = np.where(price_changes>0,price_changes,0.0).mean()
    prev_loss = np.where(price_changes<0,np.abs(price_changes),0.0).mean()
    
    # Initialize results with nans and the first RS value.
    results = np.array([*np.repeat(np.nan,periods-1),prev_gain/prev_loss])
    
    # Our period is increased by 1 because we do a first differnce for RSI,
    # hence we need period+1 datapoints.
    index = 1
    while index + periods - 1 < len(data):
        price_changes = np.diff(data[index:index+periods+1])
        mean_gain = np.where(price_changes>0,price_changes,0.0).mean()
        mean_loss = np.where(price_changes<0,np.abs(price_changes),0.0).mean()
        
        prev_gain = (prev_gain*(periods-1) + mean_gain)/periods
        prev_loss = (prev_loss*(periods-1) + mean_loss)/periods
        
        results = np.append(results,prev_gain/prev_loss)
        index += 1
    
    # Compute RSI using all of our RS calculations
    return 100 - (100/(1+results))


def stochastic_oscillator(data:np.array,period:int)->np.array:
    """Calculates the stochastic oscillator data which is a momentum indicator
    telling us whether a stock is overbought or oversold. Overbought threshold
    is set to 80 and oversold is set at 20. Usually used with the 3 period 
    moving average of itself. Implementation follows:
    https://www.investopedia.com/terms/s/stochasticoscillator.asp"""
    index = 0
    result = np.repeat(np.nan,period-1)

    while index + period-1 < data.shape[0]:
        temp = data[index:index+period]
        high = temp.max()
        low = temp.min()
        close = temp[-1]
        result = np.append(result,((close-low)/(high-low)*100))
        index += 1
    return result


def ulcer_index(data:np.array,period:int)->np.array:
    """Calculates the Ulcer index for the specified period. This is a measure
    if risk for the period. Generally when ulcer index spikes above normal 
    levels this means that a return to that price will take a long time.
    Normal levels can be ascertained from using a very long running SMA.
    Implemented from: https://www.investopedia.com/terms/u/ulcerindex.asp"""
    index = 0
    result = np.repeat(np.nan,period-1)
    while index + period-1 < data.shape[0]:
        high = data[index:index+period].max()
        close = data[index:index+period][-1]

        mini_result = np.power(
            np.multiply(
                np.divide(
                    np.subtract(close,high),high),100),2)
        result = np.append(result,np.sqrt(mini_result.mean()))
        index += 1
    return result


def accumulation_distribution_line(price_data:np.array,volume_data:np.array,period:int)->np.array:
    """The ADL measures the cummulative flow of money into and out of the system
    The main tell point is when the indicator diverges from the price.
    Implemented From:
    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line
    """
    if len(price_data) != len(volume_data):
        raise RuntimeError("Price and Volume arrays ust be the same length")

    index = 0
    results = np.repeat(np.nan,period-1)

    prev_adl = 0.0
    while index + period-1 < len(price_data):
        prices = price_data[index:index+period]
        volumes = volume_data[index:index+period]
        
        high = prices.max()
        low = prices.min()
        close = prices[-1]
        money_multiplier = ((close-low)-(high-close)) / (high-low)
        if np.isnan(money_multiplier) or np.isinf(money_multiplier):
            money_multiplier = 1.0

        adl = np.sum(volumes)*money_multiplier
        results = np.concatenate([results,np.repeat(prev_adl + adl,period)])

        prev_adl = prev_adl + adl
        index += period
    # Cut the extra repetitions
    return results[:len(price_data)]
    

def balance_of_power(data:np.array,period:int,smoothing_period:int)->np.array:
    """The Balance of power is an indicator fluctuating from -1 to 1 which 
    measures the strength of buying and selling pressure. When postive bulls
    are in charge, when negative bears. Typically smoothed using the smoothing
    period. Implemented from:
    https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
    """
    index = 0
    results = np.repeat(np.nan,period-1)
    # Because indexing is exclusive of the end we do a -1
    while index + period - 1 < len(data):
        temp = data[index:index+period]
        bop = (temp[-1]-temp[0]) / (temp.max()-temp.min())
        if np.isnan(bop) or np.isinf(bop):
            bop = 0.0
        results = np.append(results,bop)
        index += 1
    
    if smoothing_period > 1:
        return simple_moving_average(results,smoothing_period)
    return results


def average_true_rating(data:np.array,window:int,period:int)->np.array:
    """Average true rating is exclusively a measure of volatility. The higher
    the graph moves, the higher the volatility it is indicating. Note since
    this is an absolute measure, high price stocks will have higher ATR and
    visa versa. Reveals the degree of interest or disinterest in a move.
    Implemented from:
    https://school.stockcharts.com/doku.php?id=technical_indicators:average_true_range_atr
    
    Parameters:
        window (int): The number of datapoints to use per Kline. This is the 
            span from which we take our high and low values.
        period (int): The smoothing factor for our ATR."""
    # First we calculate the high-low for each window period
    index = 0
    true_rating = np.array([],dtype=np.float64)
    while index + window < len(data):
        temp = data[index:index+window]
        true_rating = np.append(true_rating,temp.max()-temp.min())
        index += 1
    # The first Average true rating is just the mean of the first 14 TRs.
    atr = np.repeat(np.nan,window-1)
    atr = np.append(atr,true_rating[:period].mean())
    
    prev_atr = atr[-1]
    for i in range(len(true_rating)):
        atr = np.append(atr,((prev_atr*(period-1))+true_rating[i])/period)
        prev_atr = atr[-1]
    return atr