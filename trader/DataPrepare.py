from __future__ import annotations
import pandas as pd
from typing import Dict,Tuple


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