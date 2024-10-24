import re
import nltk
import torch
import tensorflow as tf
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import transformers
from transformers import BertModel
import requests
from bs4 import BeautifulSoup, Comment
import time
from sentence_transformers import SentenceTransformer
from cachetools import cached, TTLCache
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException, HTTPError

pattern = re.compile(
    r'/cart/*|/checkout/*$|'
    r'/login/*|/signup/*|/log-in/*|/account/*|/my-account/*|/register/*'
    r'/return-policy/*|/privacy-policy/*|/terms-of-service/*|/refund-policy/*|/shipping-policy/*|/terms-conditions'
    r'/*|/terms-and-conditions/*|'
    r'/reviews/*|/all-reviews/*|'
    r'/contact-us/*|/contact/*$|/contactus/*|/customer-service/*|'
    r'/aboutus/*$|/about-us/*|/about/*|'
    r'/index.php/*$|/home/*|'
    r'/blog/*|/blogs/*|/news/*|/our-brands/*$|'
    r'/search/*$|/merch/*$|/donate/*|/faq/*|/forums/*|'
    r'/collections/*|/category/*|/product-category/*|/product-brands/*'
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}

cache = TTLCache(maxsize=1000, ttl=600)
nltk.download('punkt')
executor = ThreadPoolExecutor(max_workers=10)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # paraphrase-MiniLM-L6-v2 all-MiniLM-L6-v2
# TODO plan: take product and non-product webpage from stage 2, embed them and use them to compare with new embeddings


def createStopWordsSet():
    print('## NOW RUNNING createStopWordsSet')
    stop_words = set(stopwords.words('english'))
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


@cached(cache)
def fetchVisibleContent(url):
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        for hidden in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
            hidden.extract()

        visible_text = soup.get_text(separator=' ')
        exception_counters['successful'] += 1
        return ' '.join(visible_text.split())

    except requests.Timeout:
        print(f"Timeout error, url={url}")
        exception_counters['timeout_error'] += 1
        return ''
    except requests.HTTPError:
        print(f"HTTP error, url={url}")
        exception_counters['http_error'] += 1
        return ''
    except requests.exceptions.InvalidURL:
        print(f"Invalid url, url={url}")
        exception_counters['invalidURL_error'] += 1
        return ''
    except requests.RequestException:
        print(f"Request Exception, url={url}")
        exception_counters['request_error'] += 1
        return ''
    except Exception as e:
        print(f"Error occurred, url={url}, e={e} ")
        exception_counters['unknown_error'] += 1


def fetchVisibleContent_Wrapper(df):
    print('## NOW RUNNING fetchVisibleContent_Wrapper')
    # data_filtered['Content'] = data_filtered['sublinks'].apply(fetchVisibleContent)
    df['Content'] = list(executor.map(fetchVisibleContent, df['sublinks']))


def preprocessText(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    if stop_words_set is not None:
        lst_text = [word for word in lst_text if word not in
                    stop_words_set]

    lst_text = [ps.stem(word) for word in lst_text]
    lst_text = [lem.lemmatize(word) for word in lst_text]
    text = " ".join(lst_text)
    return text


def preprocessText_Wrapper(df):
    print('## NOW RUNNING preprocessText_Wrapper')
    df['Content_clean'] = df['Content'].apply(lambda x: preprocessText(x))
    df.drop(columns=['Content'], inplace=True)


def splitTextIntoChunks(text, max_tokens=300):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(model.tokenize(sentence))
        # If adding the next sentence exceeds max_tokens, start a new chunk
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    # Add the last chunk if there is any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def generateAverageEmbedding(text):
    # Split text into manageable chunks
    chunks = splitTextIntoChunks(text)
    # Generate embeddings for each chunk
    chunk_embeddings = model.encode(chunks)
    # Compute the mean of the chunk embeddings
    average_embedding = np.mean(chunk_embeddings, axis=0)
    return average_embedding


def generateAverageEmbedding_Wrapper(df):
    df['Avg_embed'] = df['Content_clean'].apply(lambda x: generateAverageEmbedding(x))
    df.drop(columns=['Content_clean'], inplace=True)


if __name__ == '__main__':
    start_time = time.time()
    exception_counters = defaultdict(int)

    classified_data = pd.read_csv('../output/productpage_classification_based_regex_dataset1_2024-10-16 18-25.csv')
    product_data = classified_data[classified_data['Product Page'] == 1].copy()
    # TODO filter out only what's 100% non product page
    non_product_data = classified_data[classified_data['Product Page'] == 0].copy()

    # TODO compare data_to_classify with product and non-product embeddings.

    stop_words_set = createStopWordsSet()
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()

    # fetch visible content for Product Webpages and Non-Product Webpages from dataset 1 stage 2
    fetchVisibleContent_Wrapper(df=product_data)
    fetchVisibleContent_Wrapper(df=non_product_data)

    # preprocess webpage content of Product Webpages and Non-Product Webpages
    preprocessText_Wrapper(df=product_data)
    preprocessText_Wrapper(df=non_product_data)

    generateAverageEmbedding_Wrapper(df=product_data)
    generateAverageEmbedding_Wrapper(df=non_product_data)

    data_to_classify = pd.read_parquet('./data/parquet_output/sublinks_depth7_2024-10-14 16-43.parquet')
    fetchVisibleContent_Wrapper(df=data_to_classify)
    preprocessText_Wrapper(df=data_to_classify)
    generateAverageEmbedding_Wrapper(df=data_to_classify)








    # TODO fetch content
    # TODO preprocess
    # TODO run embeddings on product_data and non_product_data
    # TODO run embeddings on new data ( Parquet files )
    # TODO multiply the embeddings of the new data with the Product/Non_Product data and classify according to distance
