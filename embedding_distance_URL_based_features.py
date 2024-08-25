from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams, bigrams
import nltk
import re
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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


def removeStopWordsWrapper(df, column_name):
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


def makeLowerCase(df):
    def lowerCase(txt):
        return txt.lower()

    df['sublinks'] = df['sublinks'].apply(lowerCase)
    return df


def prepareDataFrame():
    global sublinks, stop_words_g
    sublinks = makeLowerCase(sublinks)
    sublinks = remove_www(sublinks)
    sublinks = standardize_url_wrapper(sublinks)
    sublinks = extractAfterTldWrapper(sublinks)
    stop_words_g = createStopWordsSet()
    sublinks = removeStopWordsWrapper(sublinks, 'after_tld')
    sublinks = handleMissingValues(sublinks)


def preprocess_after_tld(df):
    def preprocess_txt(txt):
        if not txt:
            return ""

    # TODO lemmatization of words , tokens df[preprocessed_after_tld] = ...
    return df



if __name__ == '__main__':
    sublinks = pd.read_csv('data/sublinks.csv')
    stop_words_g = set()
    prepareDataFrame()
    sublinks = preprocess_after_tld(sublinks)
    #TODO create embeddings
    sublinks.to_csv('output/embeddings_stage2_data_prep.csv', index=False)
