from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict,Tuple,List


def prepare_raw_data(data_path:str,config_data:Dict)->pd.DataFrame:
    """Read in the specified data and select only the variables used in the
    in the creation of trees. Also clean up the index column if it exists
    Parameters:
        config_data (Dict): Configuration data for this run."""
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0",inplace=True,axis=1)

    if "variables" in config_data:
        return df[config_data["variables"]]
    else:
        return df



def continuos_train_test_split(data:pd.DataFrame,split:float)->Tuple:
    """Generate a continuous train/test split of data. Meaning we do not sample
    the data, but rather simply split it at the integer index corresponding to
    the split percent specified.
    Parameters:
        split (float): The percent of data to be included in the training set.
    Returns a tuple of (train_dataframe, test_dataframe)"""
    split_point = int(len(data.index) * split)

    train = data.iloc[:split_point].copy()
    test = data.iloc[split_point:].copy()

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
    
    print(F"\tSampling data from [{split_point}:{split_offset}]")

    sampled_data = data.iloc[split_point:split_offset].copy()
    return sampled_data


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
