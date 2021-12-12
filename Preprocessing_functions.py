# this is the file with useful functions in data preprocessing
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import numpy as np




def safe_drop_attr(df, label, list_drop):
    """
    This function is just a variation of the usual drop, we are using it just for clarity
    :param df: Pandas dataframe
    :param list_drop: list containing the labels to be dropped
    :param label: label to drop
    :return: the df without the given label
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a DF!")

    # drop the label contained in list_drop
    for i in list_drop:
        df[i].drop(inplace=True)

    return df


def safe_eliminate_NaN(df):
    """
    This function eliminates rows containing NaN values from the given DF. It also tells how many features were
    eliminated in that process
    :param df: pandas DataFrame
    :return: the df without NaN values
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a DF!")

    inst_before = df.shape[0]
    # drop rows containing NaN values
    df.dropna()
    inst_after = df.shape[0]
    print("The number of instances after dropped are: ", inst_before - inst_after)

    return df



