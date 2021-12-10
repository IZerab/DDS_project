# this is the file with useful functions in data preprocessing
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import numpy as np


def drop_non_geolocalised(df, label_latitude, label_longitude):
    """
    This function selects only the elements of the df that are geolocalised
    :param df: dataframe with geolocalised data
    :param label_latitude: feature name of the latitude
    :param label_longitude: feature name of the longitude
    :return: the df with only geolocalized data
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a DF!")

    num_df = df.shape[0]
    print("The number of instances in the df is: ", num_df)

    # we remove the data we are not able to localize!
    # first we drop latitude
    df = df[df[label_latitude].notna()]

    # then we drop longitude
    df = df[df[label_longitude].notna()]

    num_df = df.shape[0]
    print("The number of instances after dropping the non localized records is: ", num_df)

    return df

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


def localize_tweets(df, title):
    """
    Function that creates a shapely like object to be use to gelocalize the single tweets using geopandas
    :param df: Our dataframe with coordinates (lat and long). It plots te results on the world map!
    :param title: title of the plot
    :return: the geopandas dataframe with the geolocalized data!
    """
    # creating the geopandas dataset
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.long, df.lat))

    # importing world map
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # get the ID of each state
    #categories = np.unique(gdf["id"])
    #colors = np.linspace(0, 1, len(categories))
    #colordict = dict(zip(categories, colors))

    # adding the color to each record
    #gdf["Color"] = gdf["id"].apply(lambda x: colordict[x])

    # plot the world map
    ax = world.plot(color='white', edgecolor='black')
    ax.set_title(title)

    # plot our points
    gdf.plot(ax=ax, marker=',', markersize=1, cmap='BuGn')

    plt.show()

    return gdf
