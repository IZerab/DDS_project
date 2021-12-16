# this lib contains NLP functions

# lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


# Sentiment analysis
# Language detection using spacy¶
# More details here: https://spacy.io/universe/project/spacy-langdetect
def clean(text):
    '''This function takes a string and cleans it from anything that is not a character also it lowers all characters.
    :param text: the string of text that should be cleaned
    :return: the cleaned text
    '''
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', str(text))
    return text


@Language.factory("language_detector")
# define the function
def get_lang_detector(nlp, name):
    '''This function finds the language of a given text
    :param nlp: the language model
    :param name: the class object
    :return: the language of the text
    '''
    return LanguageDetector()


def get_the_lenguages(df):
    """
    take the cleaned tweets and create an empty list, then loops through tweets and add language to the list
    :param df: dataframe we are working on
    :return: the list of the lenguages in the twetter database
    """
    tweets = df['cleantweet']
    languages_spacy = []

    for e in tweets:
        doc = nlp(e)
        # cheking if the doc._.languages is not empty
        # then appending the first detected language in a list
        if doc._.language:
            print(doc._.language['language'])
            languages_spacy.append(doc._.language['language'])
        else:
            print("NaN")
            languages_spacy.append('NaN')

        return languages_spacy
