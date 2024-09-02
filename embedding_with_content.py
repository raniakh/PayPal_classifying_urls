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
import requests

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def makeLowerCase(df):
    def lowerCase(txt):
        return txt.lower()

    df['sublinks'] = df['sublinks'].apply(lowerCase)
    return df


def remove_www(df, column_name='sublinks'):
    df[column_name] = df[column_name].str.replace('www.', '', regex=False)
    return df


def standardize_url_wrapper(df, column_name='sublinks'):
    def standardize_url(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        path = parsed_url.path
        if path == '/' or path == '':
            return f'{scheme}://{domain}'
        return url

    df[column_name] = df[column_name].apply(standardize_url)
    return df


def extractAfterTldWrapper(df, column_name='sublinks'):
    def extractAfterTld(url):
        parsed_url = urlparse(url)
        after_tld = parsed_url.path
        if parsed_url.params:
            after_tld += ';' + parsed_url.params
        if parsed_url.query:
            after_tld += '?' + parsed_url.query
        if parsed_url.fragment:
            after_tld += '#' + parsed_url.fragment
        if not after_tld or after_tld == " ":  # if this is a homepage
            after_tld = 'None'
        return after_tld

    df['after_tld'] = df[column_name].apply(extractAfterTld)

    return df


def createStopWordsSet():
    stop_words = set(stopwords.words('english'))
    url_top_words = ["pdf", "html"]  # TODO: check for more stop words - analysis
    stop_words.update(url_top_words)
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


def remove_specialCharsAndDigits(df, column_name):
    def clean_url(url):
        pattern = r'[\d_\.\-\:\?\$\%\/\=\+]'
        return re.sub(pattern, ' ', url)

    df[column_name] = df[column_name].apply(clean_url)
    return df


def tokenizeTxt(txt):
    tokens = word_tokenize(txt)
    return ' '.join(tokens)


def removeSpecialCharsAndStopWordsWrapper(df, column_name):
    def removeStopWords(txt):
        words = txt.split()
        filtered_words = [word for word in words if word not in stop_words_g and len(word) > 1]
        return ' '.join(filtered_words)

    df = remove_specialCharsAndDigits(df, column_name)
    df[column_name] = df[column_name].apply(tokenizeTxt)
    df[column_name] = df[column_name].apply(removeStopWords)

    return df


def handleMissingValues(df):
    df['after_tld'] = df['after_tld'].replace("", "None")
    return df


def preprocessAfterTLD(df):
    wnl = WordNetLemmatizer()

    def preprocess_txt(txt):
        if not txt:
            return ""
        tokens = word_tokenize(txt)
        tokens = [wnl.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    df['processed_after_tld'] = df['after_tld'].apply(preprocess_txt)
    return df


def extractMetadata(url):
    pass


def prepareDataFrame():
    global sublinks, stop_words_g
    sublinks = makeLowerCase(sublinks)
    sublinks = remove_www(sublinks)
    sublinks = standardize_url_wrapper(sublinks)
    sublinks = extractAfterTldWrapper(sublinks)
    stop_words_g = createStopWordsSet()
    sublinks = removeSpecialCharsAndStopWordsWrapper(sublinks, 'after_tld')
    sublinks = handleMissingValues(sublinks)


if __name__ == '__main__':
    start_time = time.time()
    sublinks = pd.read_csv('data/sublinks.csv')
    sublinks = sublinks.iloc[:50]
    stop_words_g = set()
    prepareDataFrame()

# TODO IDEAS: map tld to type ans use as a feature?
# TODO -> https://onlinelibrary.wiley.com/doi/10.1155/2021/2470897 ->
#  this paper designs an algorithm to identify noisy text tags,
#  features: Title tag, Description tag,
