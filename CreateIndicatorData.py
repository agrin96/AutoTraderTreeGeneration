from Indicators import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("INDICATOR GENERATOR")

def generate_technical_indicators(
        df:pd.DataFrame,
        price_column:str,
        volume_column:str,
        volume_seed:float=10000.0):
    
    logger.info("Generating metadata.")
    price_data = df[price_column].values
    volume_data = df[volume_column].values
    # Sice we have only total traded volume in a 24 hour period. We take a
    # first difference and use a seed first volume to get by second volume
    volume_data = np.cumsum(np.insert(np.diff(volume_data),0,volume_seed))

    column_names = ["volume"]
    output_data = np.array(volume_data).reshape(1,-1)

    # Any time the kline period is multiplied this is to generate some type of
    # moving average or a calculation period. This is because our data is
    # by second.
    kline_period = 60

    #------------------------------------------------------------------------#
    logger.info("Generate Average True Rating")
    atr = average_true_rating(
        data=price_data,
        window=kline_period,
        period=14)
    atr_sma = simple_moving_average(atr,60*3)
    
    column_names.extend(["atr","atr_sma"])
    output_data = np.append(output_data,atr.reshape(1,-1),axis=0)
    output_data = np.append(output_data,atr_sma.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    # Balance of power, its SMA, and the derivative of the SMA
    logger.info("Generate Balance of Power")
    bop = balance_of_power(
        data=price_data,
        period=kline_period,
        smoothing_period=kline_period*14)
    bop_sma = simple_moving_average(bop,60*3)
    ddx_degree = 3
    ddx_bop = np.subtract(bop_sma,array_shift(bop_sma,ddx_degree))/ddx_degree
    
    column_names.extend(["bop","bop_sma","bop_prime"])
    output_data = np.append(output_data,bop.reshape(1,-1),axis=0)
    output_data = np.append(output_data,bop_sma.reshape(1,-1),axis=0)
    output_data = np.append(output_data,ddx_bop.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate Accumulatrion Distribution Line")
    adl = accumulation_distribution_line(
        price_data=price_data,
        volume_data=volume_data,
        period=30)
    adl = rescale(arr=adl,low=price_data.min(),high=price_data.max())
    column_names.append("adl")
    output_data = np.append(output_data,adl.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate Ulcer Index")
    ulcer = ulcer_index(data=price_data,period=kline_period*2)
    ulcer_normal = simple_moving_average(ulcer,kline_period*2*8)
    ulcer_diff = ulcer_normal-ulcer
    
    column_names.extend(["ulcer","ulcer_normal","ulcer_diff"])
    output_data = np.append(output_data,ulcer.reshape(1,-1),axis=0)
    output_data = np.append(output_data,ulcer_normal.reshape(1,-1),axis=0)
    output_data = np.append(output_data,ulcer_diff.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate Stochastic Oscillator")
    stochastic = stochastic_oscillator(
        data=price_data,
        period=kline_period*10)
    stochastic_signal = simple_moving_average(stochastic,kline_period*2)
    stochastic_diff = stochastic_signal-stochastic

    column_names.extend(["stochastic","stochastic_signal","stochastic_diff"])
    output_data = np.append(output_data,stochastic.reshape(1,-1),axis=0)
    output_data = np.append(output_data,stochastic_signal.reshape(1,-1),axis=0)
    output_data = np.append(output_data,stochastic_diff.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate Relative Strength Index")
    rsi = relative_strength_index(data=price_data,periods=28)
    rsi_oversold = np.repeat(30,len(rsi)) - rsi
    rsi_overbought = rsi - np.repeat(70,len(rsi))

    column_names.extend(["rsi","rsi_oversold","rsi_overbought"])
    output_data = np.append(output_data,rsi.reshape(1,-1),axis=0)
    output_data = np.append(output_data,rsi_oversold.reshape(1,-1),axis=0)
    output_data = np.append(output_data,rsi_overbought.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate MACD")
    macd_line,signal,histogram = mean_average_convergance_divergance(
        data=price_data,
        fast_window=kline_period*4,
        slow_window=kline_period*9,
        signal_window=kline_period*3)
    
    column_names.extend(["macd_4_9_3","macd_4_9_3_signal"])
    output_data = np.append(output_data,macd_line.reshape(1,-1),axis=0)
    output_data = np.append(output_data,signal.reshape(1,-1),axis=0)

    macd_line,signal,histogram = mean_average_convergance_divergance(
        data=price_data,
        fast_window=kline_period*12,
        slow_window=kline_period*26,
        signal_window=kline_period*9)

    column_names.extend(["macd_12_26_9","macd_12_26_9_signal"])
    output_data = np.append(output_data,macd_line.reshape(1,-1),axis=0)
    output_data = np.append(output_data,signal.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate Bollinger Bands")
    upper,middle,lower,bandwidth,percentB = bollinger_bands(
        data=price_data,
        window=kline_period*20)

    column_names.extend(["bollinger_upper","bollinger_lower","percent_b"])
    output_data = np.append(output_data,upper.reshape(1,-1),axis=0)
    output_data = np.append(output_data,lower.reshape(1,-1),axis=0)
    output_data = np.append(output_data,percentB.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    logger.info("Generate On Balance Volume")
    obv = on_balance_volume(data=volume_data,period=10)

    column_names.append("obv")
    output_data = np.append(output_data,obv.reshape(1,-1),axis=0)
    
    #------------------------------------------------------------------------#
    logger.info("Generate Adaptive Moving Average")
    kama_short = kaufman_adaptive_moving_average(
        data=price_data,
        er_period=kline_period*10,
        fast_ema=2,
        slow_ema=30)

    kama_mid = kaufman_adaptive_moving_average(
        data=price_data,
        er_period=kline_period*10,
        fast_ema=2*3,
        slow_ema=30*3)

    kama_long = kaufman_adaptive_moving_average(
        data=price_data,
        er_period=kline_period*10,
        fast_ema=2*6,
        slow_ema=30*6)

    column_names.extend(["kama_short","kama_mid","kama_long"])
    output_data = np.append(output_data,kama_short.reshape(1,-1),axis=0)
    output_data = np.append(output_data,kama_mid.reshape(1,-1),axis=0)
    output_data = np.append(output_data,kama_long.reshape(1,-1),axis=0)

    #------------------------------------------------------------------------#
    return pd.DataFrame(output_data.T,columns=column_names)

def main():
    raw_data = pd.read_csv(f"./ticker.csv")
    indicators = generate_technical_indicators(
        df=raw_data,
        price_column="best_ask",
        volume_column="total_traded_asset")
    work_data = pd.concat(
        [raw_data[["timestamp","best_ask","best_bid"]],
        indicators],axis=1)
    work_data.dropna(how='any',inplace=True)
    work_data.to_csv("./indicator_data.csv")

if __name__ == "__main__":
    main()