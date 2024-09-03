from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
import re
import numpy as np
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def createStopWordsSet():
    stop_words = set(stopwords.words('english'))
    url_top_words = ["pdf", "html"]  # TODO: check for more stop words - analysis
    stop_words.update(url_top_words)
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


def remove_specialCharsAndDigits(column_name):
    global sublinks

    def clean_url(url):
        pattern = r'[\d_\.\-\:\?\$\%\/\=\+\,\—\–\"\``]'
        return re.sub(pattern, ' ', url)

    sublinks[column_name] = sublinks[column_name].apply(clean_url)


def tokenizeTxt(txt):
    tokens = word_tokenize(txt)
    return ' '.join(tokens)


def removeSpecialCharsAndStopWordsWrapper(column_name):
    global sublinks

    def removeStopWords(txt):
        words = txt.split()
        filtered_words = [word for word in words if word not in stop_words_g and len(word) > 1]
        return ' '.join(filtered_words)

    remove_specialCharsAndDigits(column_name)
    sublinks[column_name] = sublinks[column_name].apply(tokenizeTxt)
    sublinks[column_name] = sublinks[column_name].apply(removeStopWords)


def handleMissingValues():
    global sublinks
    # df['after_tld'] = df['after_tld'].replace("", "None")
    for column_name in sublinks:
        sublinks[column_name] = sublinks[column_name].fillna("None")


def extractMetadata(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string if soup.title else 'None'

        description = None
        if soup.find("meta", attrs={"name": "description"}):
            description = soup.find("meta", attrs={"name": "description"}).get("content")
        if not description:
            description = 'None'
        return {'sublinks': url, 'title': title, 'description': description}
    except (requests.exceptions.RequestException, Exception) as e:
        print("Exception occurred while requesting {url}", url)
        print("#Error: ", str(e))
        return {'sublinks': url, 'title': None, 'description': None}


def extractMetadataParallel(urls, max_wrokers=10):
    results = []

    with ThreadPoolExecutor(max_workers=max_wrokers) as executor:
        future_to_url = {executor.submit(extractMetadata, url): url for url in urls}

        for future in as_completed(future_to_url):
            try:
                result = future.result()
                results.append(result)
            except Exception as ex:
                print('#Error: ', str(ex))
                url = future_to_url[future]
                results.append({'sublinks': url, 'title': None, 'description': None})
    return results


def extractMetadataParallelWrapper():
    global sublinks
    results = extractMetadataParallel(sublinks['sublinks'].tolist(), max_wrokers=10)
    df_metadata = pd.DataFrame(results)
    sublinks = sublinks.merge(df_metadata, on='sublinks')


def preprocessMetaData():
    global sublinks
    wnl = WordNetLemmatizer()

    def preprocess_txt(txt):
        if not txt:
            return ""
        tokens = word_tokenize(txt)
        tokens = [wnl.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    sublinks['title'] = sublinks['title'].apply(preprocess_txt)
    sublinks['description'] = sublinks['description'].apply(preprocess_txt)


def getEmbeddingsWrapper():
    global sublinks

    def getEmbeddings(txt, model):
        words = txt.split()
        word_embeddings = [model[word] for word in words if word in model]
        if word_embeddings:
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros(model.vector_size)

    word_vectors = api.load("word2vec-google-news-300")
    sublinks['embeddings_title'] = sublinks['title'].apply(lambda x: getEmbeddings(x, word_vectors))
    sublinks['embeddings_desc'] = sublinks['description'].apply(lambda x: getEmbeddings(x, word_vectors))


def combineEmbeddings():
    global sublinks

    sublinks['combined_embeddings'] = sublinks.apply(lambda row: row['embeddings_after_tld'] + row['embeddings_title']
                                                     + row['embeddings_desc'], axis=1)

    # combined_embed = sublinks['combined_embeddings'].tolist()
    # scaler = StandardScaler()
    # combined_embed_scaled = scaler.fit_transform(combined_embed)
    # return combined_embed_scaled


def cluster(n_clusters=10):
    global sublinks
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    embeddings = np.array(sublinks['combined_embeddings'].tolist())
    classes = kmeans.fit_predict(embeddings)
    sublinks['cluster_stage3'] = (list(map(str, classes)))


if __name__ == '__main__':
    start_time = time.time()
    # sublinks = pd.read_csv('output/embeddings_stage2.csv')
    sublinks = pd.read_pickle('output/embeddings_stage2.pkl')
    # sublinks = sublinks.iloc[:50]
    extractMetadataParallelWrapper()
    handleMissingValues()
    stop_words_g = createStopWordsSet()
    removeSpecialCharsAndStopWordsWrapper('title')
    removeSpecialCharsAndStopWordsWrapper('description')
    preprocessMetaData()
    getEmbeddingsWrapper()
    combineEmbeddings()
    cluster(n_clusters=10)

    sublinks.to_csv('output/sublinks_with_content_combine_embed.csv', index=False)
    sublinks.to_pickle('output/sublinks_with_content_combine_embed.pkl')

# TODO IDEAS: map tld to type ans use as a feature?
# TODO -> https://onlinelibrary.wiley.com/doi/10.1155/2021/2470897 ->
#  this paper designs an algorithm to identify noisy text tags,
#  TODO -> create embeddings for title and description.
#            merge with after_tld embeddings.
#             Cluster
#
