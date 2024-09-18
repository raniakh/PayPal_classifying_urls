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


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

######
# STAGE 2 OF THE FUNNEL - POC SHOWED IT WORKED. 
######
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
    sublinks = removeSpecialCharsAndStopWordsWrapper(sublinks, 'after_tld')
    sublinks = handleMissingValues(sublinks)


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


def getEmbeddingsWrapper(df):
    def getEmbeddings(txt, model):
        words = txt.split()
        word_embeddings = [model[word] for word in words if word in model]
        if word_embeddings:
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros(model.vector_size)

    word_vectors = api.load("word2vec-google-news-300")
    df['embeddings_after_tld'] = df['processed_after_tld'].apply(lambda x: getEmbeddings(x, word_vectors))
    return df


def calculateSimilarity(df):
    similarity_matrix = cosine_similarity(np.vstack(df['embeddings_after_tld'].values))
    return similarity_matrix


def cluster(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    embeddings = np.array(df['embeddings_after_tld'].tolist())
    classes = kmeans.fit_predict(embeddings)
    df['cluster_stage2'] = (list(map(str, classes)))
    return df


def plotClusters(df):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings = np.array(df['embeddings_after_tld'].tolist())
    embeddings2d = tsne.fit_transform(embeddings)

    scatter = plt.scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=df['cluster_stage2'].astype(int))
    plt.colorbar(scatter, label='Cluster')
    plt.title('2D Scatter Plot of URLs Colored by Cluster')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    sublinks = pd.read_csv('data/sublinks.csv')
    sublinks = sublinks.iloc[:300]
    stop_words_g = set()
    prepareDataFrame()
    sublinks = preprocessAfterTLD(sublinks)
    sublinks.to_csv('output/embeddings_stage2_data_prep.csv', index=False)
    sublinks = getEmbeddingsWrapper(sublinks)
    sublinks = cluster(sublinks, n_clusters=20)
    plotClusters(sublinks)
    header = ["sublinks", "after_tld", "cluster_stage2"]
    sublinks.to_csv('output/embeddings_stage2_20Clusters.csv', columns=header, index=False)
    sublinks.to_pickle('output/embeddings_stage2.pkl')
    print("--- %.2f seconds ---" % (time.time() - start_time))
