# this is the main of the project

# libraries
import numpy as np
import pandas as pd
import spacy
import geopandas
import sklearn as sk
import matplotlib.pyplot as plt

# custom libraries
from Geographycal_functions import drop_non_geolocalised
from Geographycal_functions import localize_tweets
from NLP_functions import clean
from NLP_functions import get_the_lenguages

# import the raw data
data_donald = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
data_joe = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')

# create the text mined features
# load english
nlp = spacy.load("en_core_web_sm")
# this is just an example of how this can look like
nlp.add_pipe('language_detector', last=True)
print(nlp("This is an english text.")._.language)

data_donald['cleantweet'] = data_donald['tweet'].apply(clean)
data_joe['cleantweet'] = data_joe['tweet'].apply(clean)

# get rid of the text we don't need # inplace=True changes the thing in the data set
data_donald['tweet'].drop(axis=1, inplace=True)
data_joe['tweet'].drop(axis=1, inplace=True)

# find the list of the leanguages the tweets where written in
languages_donald = get_the_lenguages(data_donald)
languages_joe = get_the_lenguages(data_joe)

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
