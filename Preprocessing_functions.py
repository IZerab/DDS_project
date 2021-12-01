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
