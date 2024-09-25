from urllib.parse import urlparse, urlunparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk
import re
import requests
import numpy as np
import pandas as pd
import time


####
# DATA FRAME PREP FILE
####

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def makeLowerCase(column_name='sublinks'):
    global sublinks

    def lowerCase(txt):
        return txt.lower()

    sublinks[column_name] = sublinks[column_name].apply(lowerCase)


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


def extractURLcomponents(inputcol='sublinks'):
    global sublinks
    sublinks['domain'] = sublinks[inputcol].apply(lambda x: urlparse(x).netloc)
    sublinks['path'] = sublinks[inputcol].apply(lambda x: urlparse(x).path)
    # sublinks['params'] = sublinks[inputcol].apply(lambda x: urlparse(x).params)
    # sublinks['query'] = sublinks[inputcol].apply(lambda x: urlparse(x).query)
    sublinks['fragment'] = sublinks[inputcol].apply(lambda x: urlparse(x).fragment)


def createStopWordsSet():
    stop_words = set(stopwords.words('english'))
    url_top_words = ["pdf", "html"]  # TODO: check for more stop words - analysis
    stop_words.update(url_top_words)
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


def remove_specialCharsAndDigits(column_name):
    def clean_url(url):
        pattern = r'[\d_\.\-\:\?\$\%\/\=\+]'
        return re.sub(pattern, ' ', url)

    sublinks[column_name] = sublinks[column_name].apply(clean_url)


def tokenizeTxt(txt):
    tokens = word_tokenize(txt)
    return ' '.join(tokens)


def removeSpecialCharsAndStopWordsWrapper(column_names=('path', 'fragment')):
    def removeStopWords(txt):
        words = txt.split()
        filtered_words = [word for word in words if word not in stop_words_g and len(word) > 1]
        return ' '.join(filtered_words)

    for colname in column_names:
        remove_specialCharsAndDigits(colname)
        sublinks[colname] = sublinks[colname].apply(tokenizeTxt)
        sublinks[colname] = sublinks[colname].apply(removeStopWords)


def handleMissingValues():
    global sublinks
    sublinks.fillna("None", inplace=True)
    for col in sublinks.columns[1:]:
        sublinks[col] = sublinks[col].replace("", "None")


# URLS were crawled with query part - remove them
# TODO next crawl do not save the query part of the url.
def removeQuery():
    global sublinks

    def remove_query(url):
        parsed_url = urlparse(url)
        url_without_query = urlunparse(parsed_url._replace(query=''))
        return url_without_query

    sublinks['sublinks'] = sublinks['sublinks'].apply(remove_query)


def removeDuplicates():
    global sublinks

    sublinks = sublinks.drop_duplicates(subset='sublinks', keep='first').copy()


def removeFiles():
    global sublinks
    file_extensions = ('.jpg', '.jpeg', '.png', '.pdf', '.gif', '.bmp', '.tiff')
    sublinks = sublinks[~sublinks['sublinks'].str.endswith(file_extensions)]


def lemmatizeTxt():
    global sublinks
    wnl = WordNetLemmatizer()

    def preprocess_txt(txt):
        if not txt:
            return ""
        tokens = word_tokenize(txt)
        tokens = [wnl.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    for col in sublinks.columns[3:]:
        sublinks[col] = sublinks[col].apply(preprocess_txt)


def removeNonEnglishWebsites():
    global sublinks
    domains = dict()
    print('## NOW RUNNING removeNonEnglishWebsites')

    def detect_language(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        clean_url = f'{scheme}://{domain}'
        try:
            if domain in domains.keys():
                return domains.get(domain)
            head_response = requests.head(clean_url, timeout=5)
            content_type = head_response.headers.get('Content-Type', '')

            if 'text/html' in content_type:
                response = requests.get(clean_url, timeout=5, stream=True)
                response.raise_for_status()
                content_chunk = response.iter_content(chunk_size=512)
                first_chunk = next(content_chunk, b'').decode('utf-8', errors='ignore')

                if first_chunk.strip():
                    lang = detect(first_chunk)
                    domains.update({domain: lang})
                    return lang
                else:
                    domains.update({domain: 'unknown'})
                    return 'unknown'
            else:
                domains.update({domain: 'unknown'})
                return 'unknown'
        except (requests.exceptions.RequestException, LangDetectException):
            domains.update({domain: 'unknown'})
            return 'unknown'

    sublinks['language'] = sublinks['sublinks'].apply(detect_language)
    sublinks = sublinks[sublinks['language'] == 'en']


def urlLength():
    global sublinks
    sublinks['length'] = sublinks['sublinks'].apply(len)


def prepareDataFrame():
    global sublinks, stop_words_g
    stop_words_g = createStopWordsSet()
    makeLowerCase()
    removeWWW()
    removeQuery()
    standardizeUrlWrapper()
    removeDuplicates()
    removeNonEnglishWebsites()
    removeFiles()
    urlLength()
    extractURLcomponents()
    handleMissingValues()
    removeSpecialCharsAndStopWordsWrapper()
    handleMissingValues()


def preprocessDataFrame():
    lemmatizeTxt()
    sublinks.to_csv('data/sublinks_components_depth7.csv', index=False)


if __name__ == '__main__':
    start_time = time.time()
    sublinks = pd.read_csv('../data/sublinks_depth7.csv')
    # lastrows = sublinks.iloc[-50:]
    # sublinks = sublinks.iloc[:30]
    # sublinks = pd.concat([sublinks, lastrows], ignore_index=True)
    prepareDataFrame()
    preprocessDataFrame()
    print("--- %.2f seconds ---" % (time.time() - start_time))
