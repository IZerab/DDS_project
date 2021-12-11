# this is the main of the project

# libraries
import numpy as np
import pandas as pd
import geopandas
import sklearn as sk
import matplotlib.pyplot as plt

# custom libraries
from Preprocessing_functions import drop_non_geolocalised
from Preprocessing_functions import localize_tweets

# import the raw data
data_donald = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
data_joe = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')

# preprocessing the data
# data to drop
to_be_delete = ["tweet_id", "source", "user_id", "user_join_date", "user_location",
                "city", "country", "continent", "state", "state_code", "collected_at"]

# geolocalize Trump
print("Donald Trump")
data_donald = drop_non_geolocalised(data_donald, "lat", "long")

# geolocalize Joe
print("Joe Biden")
data_joe = drop_non_geolocalised(data_joe, "lat", "long")

# plotting Trump
geo_donald = localize_tweets(data_donald, "World Trump data distribution")

# plotting Biden
geo_biden = localize_tweets(data_donald, "World Joe data distribution")
