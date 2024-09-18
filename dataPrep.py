from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
import re
import numpy as np
import pandas as pd
import time


####
# DATA FRAME PREP FILE
####
def makeLowerCase():
    global sublinks

    def lowerCase(txt):
        return txt.lower()

    sublinks['sublinks'] = sublinks['sublinks'].apply(lowerCase)


def removeWWW(column_name='sublinks'):
    global sublinks
    sublinks[column_name] = sublinks[column_name].str.replace('www.', '', regex=False)


def standardizeUrlWrapper(column_name='sublinks'):
    global sublinks

    def standardize_url(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        path = parsed_url.path
        if path == '/' or path == '':
            return f'{scheme}://{domain}'
        return url

    sublinks[column_name] = sublinks[column_name].apply(standardize_url)


def extractDomainName(inputcol='sublinks', targetcol='domain'):
    global sublinks
    sublinks[targetcol] = sublinks[inputcol].apply(lambda x: urlparse(x).netloc)


def prepareDataFrame():
    global sublinks
    makeLowerCase(sublinks)
    removeWWW(sublinks)
    standardizeUrlWrapper(sublinks)
    sublinks = extractAfterTldWrapper(sublinks)
    sublinks = removeSpecialCharsAndStopWordsWrapper(sublinks, 'after_tld')
    sublinks = handleMissingValues(sublinks)


if __name__ == '__main__':
    start_time = time.time()
    sublinks = pd.read_csv('data/sublinks_depth7.csv')
