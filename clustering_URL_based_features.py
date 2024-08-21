from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams, bigrams
import re
import tldextract
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


# separators: ? query parameter , # anchor, / , & separate query parameters,
# =  assign values to parameters in the query
def calculateLengthURL(df, column_name='sublinks'):
    df['url_length'] = df[column_name].apply(len)
    return df


def remove_specialCharsAndDigits(df, column_name):
    def clean_url(url):
        pattern = r'[\d_\.\-\:\?\$\%\/\=]'
        return re.sub(pattern, ' ', url)

    df[column_name] = df[column_name].apply(clean_url)
    return df


def extractDomain(df, column_name='sublinks'):
    df['domain'] = df[column_name].apply(lambda x: urlparse(x).netloc)
    df['domain_name'] = df['domain'].apply(lambda x: x.split('.')[0])
    return df


def extractTLD(df, column_name='domain'):
    df['tld'] = df[column_name].apply(lambda x: x.split('.')[-1])
    return df


def extractSecondLevelDomainWrapper(df, column_name='domain'):
    def extractSecondLevelDomain(url):
        extracted = tldextract.extract(url)
        if extracted.suffix.count('.') > 0:
            parts = extracted.suffix.split('.')
            return parts[0] if len(parts) > 1 else None
        else:
            return None

    df['sld'] = df[column_name].apply(extractSecondLevelDomain)
    # df['sld'] = df['sld'].fillna(value="None")
    return df


def extractAfterTldWrapper(df, column_name='sublinks'):
    def extractAfterTld(url):
        parsed_url = urlparse(url)
        after_tld = parsed_url.path
        if parsed_url.query:
            after_tld += '?' + parsed_url.query
        if parsed_url.fragment:
            after_tld += '#' + parsed_url.fragment
        return after_tld

    df['after_tld'] = df[column_name].apply(extractAfterTld)
    return df


def createStopWordsSet():
    stop_words = set(stopwords.words('english'))
    url_top_words = ["pdf", "html"]  # TODO: check for more stop words - analysis
    stop_words.update(url_top_words)
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


def tokenizeTxt(txt):
    tokens = word_tokenize(txt)
    return ' '.join(tokens)


# maybe https://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
# to split concatenated words all lower case
def removeStopWordsWrapper(df, column_name):
    def removeStopWords(txt):
        words = txt.split()
        filtered_words = [word for word in words if word not in stop_words_g and len(word) > 1]
        return ' '.join(filtered_words)

    df = remove_specialCharsAndDigits(df, column_name)
    df[column_name] = df[column_name].apply(tokenizeTxt)
    df[column_name] = df[column_name].apply(removeStopWords)

    return df


def makeLowerCase(df):
    def lowerCase(txt):
        return txt.lower()

    df['sublinks'] = df['sublinks'].apply(lowerCase)
    return df


def createBiGrams(df):
    def bigrams_gen(txt):
        tokens = txt.split()
        bigrams_list = list(bigrams(tokens))
        return [' '.join(bigram) for bigram in bigrams_list]

    df['concat_txt'] = df.apply(
        lambda row: row['domain_name'] + ' ' + (row['sld'] or '') + ' ' + row['tld'] + ' ' + row['after_tld'], axis=1)
    df['bigrams'] = df['concat_txt'].apply(bigrams_gen)
    return df


def handleMissingValues(df):
    df['sld'] = df['sld'].fillna('None')
    df['after_tld'] = df['after_tld'].fillna('None')
    return df


def handleStringValues(df):
    # Options:
    # Convert strings to numerical using hashing -> sklearn.feature_extraction.text.HashingVectorizer
    #     Pros: * Efficient with large dataset, * fixed output size
    #     Cons: * Collisions (noise in features), * loss of interpretability, *no semantic info
    # Word Embeddings -> Word2Vec, GloVe
    #     Pros: * captures semantic relations, * efficient representation, *pre-trained
    #     Cons: * requires training on custom data, * limited to words (some domain names are concatenation
    #             of 2 or more words without a separator)
    # Pre-trained language model to generate embeddings for the strings -> sentence_transformers.SentenceTransformer
    #     Pros: * Powerful representations
    #     Cons: * computationally expensive, * overkill, * large in size
    # TODO: ASK : is there a way I can get a token from PayPal for an llm?

    return df


def handleBigrams(df):
    # Options
    # Bag of words sklearn.feature_extraction.text.CountVectorizer
    #     Pros: * Simple, * Interpretable, * each bigram gets its own feature
    #     Cons: * High dimensionality, * sparse features
    # TF-IDF sklearn.feature_extraction.text.TfidfVectorizer
    #     Pros: * reduces the impact of very common bigrams
    #     Cons: * high dimensionality, * sparse feature
    # Hashing Vectorizer
    #     Pros: * efficient, * scalable
    #     Cons: * hash collisions, * not interpretable
    return df


def prepareDataFrame():
    global sublinks, stop_words_g
    sublinks = makeLowerCase(sublinks)
    sublinks = remove_www(sublinks)
    sublinks = standardize_url_wrapper(sublinks)
    sublinks = calculateLengthURL(sublinks, 'sublinks')
    sublinks = extractDomain(sublinks)
    sublinks = extractTLD(sublinks)
    sublinks = extractSecondLevelDomainWrapper(sublinks)
    sublinks = extractAfterTldWrapper(sublinks)
    stop_words_g = createStopWordsSet()
    sublinks = removeStopWordsWrapper(sublinks, 'domain_name')
    sublinks = removeStopWordsWrapper(sublinks, 'after_tld')
    sublinks = createBiGrams(sublinks)


if __name__ == '__main__':
    sublinks = pd.read_csv('data/sublinks.csv')
    stop_words_g = set()
    prepareDataFrame()
    sublinks = handleMissingValues(sublinks)
    sublinks.to_csv('output/clustering_stage2_data_prep.csv', index=False)
    # TODO - stop words like "pdf" "html", find out more stop words.
    # TODO - more features might be needed.

    # 1. remove special characters in domain name except / and - => DONE
    # 2. split domain_name and after_tld into words and remove special chars from after tld => DONE
    # 3. remove stop words in domain_name and after_tld , => Done
    # 3.1.  some words in domain name are connected, should I try to separate?
    # 4. create ngrams => Done
    # 5. Clustering => in progress
    # 5.1. Feature prep: Nans, enconding, normalization => in progress
    # 5.2. hierarchical clustering
    # 5.3. saving each cluster
    # 5.4. label each cluster
