from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict,Tuple,List
from Common import arange_with_endpoint


def prepare_raw_data(data_path:str)->pd.DataFrame:
    """Read in the specified data and select only the variables used in the
    in the creation of trees. Also clean up the index column if it exists
    Parameters:
        config_data (Dict): Configuration data for this run."""
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0",inplace=True,axis=1)

    return df[["best_ask","best_bid","total_traded_asset"]]


def continuos_train_test_split(data:pd.DataFrame,split:float)->Tuple:
    """Generate a continuous train/test split of data. Meaning we do not sample
    the data, but rather simply split it at the integer index corresponding to
    the split percent specified.
    Parameters:
        split (float): The percent of data to be included in the training set.
    Returns a tuple of (train_dataframe, test_dataframe)"""
    split_point = int(len(data.index) * split)

    train = data.iloc[:split_point].copy().reset_index(drop=True)
    test = data.iloc[split_point:].copy().reset_index(drop=True)

    return train,test


def create_data_subset(data:pd.DataFrame,split:float)->pd.DataFrame:
    """Randomly choose a point in the dataframe to select a subset of the data.
    Using the split value we find out what the largest index we can select and
    still get our desired sampled dataframe size. Example:
    
    Total-length = 100, split = 0.1
    Choose a random integer in the range of (0 to (100-(100*0.1))), say 43.
    And then we return the df sliced by (43 to 43+(100*0.1))

    Returns the sliced Dataframe.
    """
    data_size = len(data.index)
    maximum_offset = data_size - int(data_size*split) - 1

    split_point = np.random.randint(0,maximum_offset)
    split_offset = split_point + int(data_size*split)
    
    print(F"\tSampling in interval [{split_point}:{split_offset}]")

    sampled_data = data.iloc[split_point:split_offset].copy()
    return sampled_data.reset_index(drop=True)


def create_data_samples(data:pd.DataFrame,
                        num_samples:int,
                        split:float)->List[pd.DataFrame]:
    """Create N number of training dataframes that we can rotate through while
    training to make sure that we generalize our bot as much as possible."""
    if num_samples == -1:
        raise RuntimeError(
        "The configuration specified means samples will be created"\
        " continuously. Therefore no fixed samples can be created.")
    
    return [create_data_subset(data,split) for _ in range(num_samples)]


def convert_ticker_to_candles(ticker:pd.DataFrame,
                              period:int=30)->pd.DataFrame:
    """Generates a dataframe of candle data using the specified options. Prices
    used are the best_asks. Volumes are determined by messing with the 24hour
    traded volume that we get each second. We diff the volumes to see how it
    changes every second and then cumsum with an initial starting volume to
    generate the volume traded at every second.
    Parameters:
        ticker (pd.DataFrame): The raw price and voluem data collected
        period (int): This is the candle stick period in seconds.
    Returns a new dataframe which only contains the candlestick data."""
    prices = ticker["best_ask"].values

    seed_volume = 100000.0 
    volume_data = np.diff(ticker["total_traded_asset"].values)
    volume_data = np.cumsum(np.insert(volume_data,0,seed_volume))

    open_idx = arange_with_endpoint(data=ticker.index.values,step=period)
    close_idx = np.add(open_idx,period-1)

    opens = prices[open_idx]
    if close_idx[-1] > prices.shape[0]:
        closes = prices[close_idx[:-1]]
        closes = np.append(closes,prices[-1])
    else:
        closes = prices[close_idx]

    highs = []
    lows = []
    volumes = []
    elements = []

    for o,c in zip(open_idx,close_idx):
        # Add one because numpy indexing excludes the last
        temp = prices[o:c+1]
        highs.append(np.max(temp))
        lows.append(np.min(temp))
        volumes.append(np.sum(volume_data[o:c+1]))
        elements.append(len(temp))

    candles = pd.DataFrame()
    candles["index"] = np.array(open_idx,dtype=np.intc)
    candles["open"] = np.array(opens,dtype=np.float64)
    candles["close"] = np.array(closes,dtype=np.float64)
    candles["low"] = np.array(lows,dtype=np.float64)
    candles["high"] = np.array(highs,dtype=np.float64)
    candles["volume"] = np.array(volumes,dtype=np.float64)
    candles["elements"] = np.array(elements,dtype=np.intc)
    return candles