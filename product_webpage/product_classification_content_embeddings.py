import re
import pandas as pd
import aiohttp
import asyncio
from datetime import datetime
from urllib.parse import urlparse
import time
import matplotlib.pyplot as plt
import nltk
from cachetools import cached, TTLCache
from joblib import Parallel, delayed
from multiprocessing import Manager
from bs4 import BeautifulSoup
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords

# TODO change input handling from csv to parquet
# Classification is not accurate, accuracu = 0, neither k-nearest neighbors nor cosine similarity if sentences.

# cache with 1000 max items and 10 minutes time-to-live
cache = TTLCache(maxsize=1000, ttl=600)

# Load SBERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # paraphrase-MiniLM-L6-v2 all-MiniLM-L6-v2

# Keywords to search for
keywords = ['buy now', 'more payment options', 'buy with apple pay', 'pay now',
            'pay with paypal', 'buy with', 'sold out', 'out of stock', 'in stock',
            'product description', 'product specifications', 'product information',
            'customer reviews', 'product reviews', 'you may also like', 'related products',
            'more from this collection', 'other products', 'request a quote', 'view store information',
            'add to cart', 'order now', 'pay now', 'purchase', 'share product']

product_phrases = [
    "buy now and add to cart",
    "this product is in stock and available for purchase",
    "customer reviews for this product",
    "you may also like other related products",
    "check more products from this collection",
    "add this product to your cart for a discount",
    "check product specifications and details",
    "limited stock available, buy now",
    "view product information and features",
    "this item is currently sold out",
    "check more payment options"
]

nltk.download('punkt')
keyword_embeddings = model.encode(keywords)
product_phrase_embeddings = model.encode(product_phrases)

nearest_neighbors_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
nearest_neighbors_model.fit(product_phrase_embeddings)

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


class URLProcessor:
    def __init__(self, exception_counters):
        self.exception_counters = exception_counters  # dictionary { exception-name: count}

    def matchRegex(self, chunk):
        print('## NOW RUNNING matchRegex')
        # Apply regex pattern filtering to eliminate non-product URLs
        chunk['regex_match'] = chunk['sublinks'].str.contains(pattern)
        chunk.loc[chunk['regex_match'], 'Product Page'] = -1  # Mark as non-product page based on regex
        return chunk.drop(columns=['regex_match'])

    def detectHomepage(self, chunk):
        print('## NOW RUNNING detectHomepage')
        # Detect if the URL is a homepage or similar and mark as non-product page
        def isHomepage(url):
            parsed_url = urlparse(url)
            return parsed_url.path in ['', '/', '/home', '/index.php']

        chunk['homepage_match'] = chunk['sublinks'].apply(isHomepage)
        chunk.loc[chunk['homepage_match'], 'Product Page'] = -1
        return chunk.drop(columns=['homepage_match'])

    @cached(cache)
    async def fetchContent(self, url, session):
        try:
            async with session.get(url, headers=headers, timeout=120) as response:
                if response.status == 200:
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        return await response.text()
                    else:
                        self._increment_exception_counter('content_type_mismatch')
                        return ''
                else:
                    self._increment_exception_counter('failed_status_code')
                    print(f"Failed to fetch {url}, status Code: {response.status}")
                    return ''
        except asyncio.TimeoutError:
            print(f"Timeout fetching {url}")
            self._increment_exception_counter('timeout_error')
            return ''
        except aiohttp.ClientConnectionError:
            print(f"Connection error fetching {url}")
            self._increment_exception_counter('connection_error')
            return ''
        except aiohttp.InvalidURL:
            print(f"Invalid URL: {url}")
            self._increment_exception_counter('invalid_url_error')
            return ''
        except UnicodeDecodeError:
            print(f"Encoding error fetching {url}")
            self._increment_exception_counter('encoding_error')
            return ''
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            self._increment_exception_counter('other_error')
            return ''

    def extractVisibleText(self, html_content):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()

            for hidden in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
                hidden.extract()

            visible_text = soup.get_text(separator=' ')
            return ' '.join(visible_text.split())
        except Exception as e:
            print(f"Error parsing HTML: {str(e)}")
            self._increment_exception_counter('html_parsing_error')
            return ''

    def chunkText(self, text, chunk_size=100):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def is_relevant_chunk(self, chunk):
        # Basic heuristic: exclude chunks with too many common stopwords
        return len(re.findall(r'\b(buy|cart|price|stock|description|review|you may also like|related '
                              r'products|product|quote|store|pay)\b', chunk)) > 0

    # Split content into sentences and classify using SBERT embeddings
    async def extractAndClassify_async(self, chunk, early_stop_threshold=0.5):
        print('## NOW RUNNING extractAndClassify_async')
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetchContent(row['sublinks'], session) for index, row in chunk.iterrows()]
            contents = await asyncio.gather(*tasks)

            for i, content in enumerate(contents):
                if content:
                    if 'globalmedicaldevices.co' in content:
                        print('debug point')
                    visible_text = self.extractVisibleText(content)
                    visible_text = visible_text.lower()
                    visible_text = preprocessText(visible_text)

                    sentences = sent_tokenize(visible_text)
                    sentence_embeddings = model.encode(sentences)

                    # chunks = self.chunkText(visible_text, chunk_size=100)
                    # chunks_filtered = [chunk for chunk in chunks if self.is_relevant_chunk(chunk)]
                    # if len(chunks_filtered) > 0:
                    #     chunks = chunks_filtered
                    #
                    # chunk_embeddings = model.encode(chunks)
                    # aggregated_embedding = np.mean(chunk_embeddings, axis=0)

                    # similarities = cosine_similarity([aggregated_embedding], keyword_embeddings)
                    # similarities = cosine_similarity([aggregated_embedding], product_phrase_embeddings)
                    ###
                    similarities = [cosine_similarity([embedding], product_phrase_embeddings) for embedding in sentence_embeddings]


                    # Apply early stopping if a highly similar sentence is found
                    max_similarity = np.max(similarities)
                    if max_similarity >= early_stop_threshold:
                        chunk.iloc[i, chunk.columns.get_loc('Product Page')] = 1
                        continue

                    if max_similarity > 0.65:  # TODO Tune the threshold
                        chunk.iloc[i, chunk.columns.get_loc('Product Page')] = 1
                    #########
                    # distances, indices = nearest_neighbors_model.kneighbors(sentence_embeddings)
                    # min_distance = np.min(distances)
                    # if min_distance <= early_stop_threshold:
                    #     chunk.iloc[i, chunk.columns.get_loc('Product Page')] = 1
                    #     continue
                    #
                    # if min_distance <= 1.1:
                    #     chunk.iloc[i, chunk.columns.get_loc('Product Page')] = 1

        return chunk

    # Wrapper to run async function in sync code
    def extractAndClassifyWrapper(self, chunk):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.extractAndClassify_async(chunk))

    def processChunk(self, chunk):
        print('## NOW RUNNING processChunk')
        chunk = self.matchRegex(chunk)
        chunk = self.detectHomepage(chunk)

        filtered_chunk = chunk[chunk['Product Page'] == 0]

        if not filtered_chunk.empty:
            filtered_chunk = self.extractAndClassifyWrapper(filtered_chunk)

        # Update the original chunk with the results
        chunk.update(filtered_chunk)
        return chunk

    def _increment_exception_counter(self, exception_type):
        """Helper method to safely increment exception counter."""
        if exception_type not in self.exception_counters:
            self.exception_counters[exception_type] = 0
        self.exception_counters[exception_type] += 1


def chunkDataframe(df, chunk_size):
    for start in range(0, df.shape[0], chunk_size):
        yield df.iloc[start: start + chunk_size]


def processDataFrameInChunks(df, chunk_size=1000, n_jobs=-1, processor=None):
    print('## NOW RUNNING processDataFrameInChunks')
    chunks = list(chunkDataframe(df, chunk_size))
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(processor.processChunk)(chunk) for chunk in chunks)

    return pd.concat(results)


def plot_exception_histogram(exception_counters_):
    print('## NOW RUNNING plot_exception_histogram')
    exception_names = list(exception_counters_.keys())
    exception_counts = list(exception_counters_.values())

    plt.bar(exception_names, exception_counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Exception Types')
    plt.ylabel('Frequency')
    plt.title('Histogram of Exceptions')
    plt.tight_layout()
    plt.show()


def createStopWordsSet():
    print('## NOW RUNNING createStopWordsSet')
    stop_words = set(stopwords.words('english'))
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


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


if __name__ == '__main__':
    start_time = time.time()

    data = pd.read_csv('../output/productpage_classification_based_regex_dataset1_2024-10-16 18-25.csv')
    data_filtered = data[data['Product Page'] == 0].copy()
    stop_words_set = createStopWordsSet()
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()

    with Manager() as manager:
        exception_counters = manager.dict(defaultdict(int))

        processor = URLProcessor(exception_counters)

        data_processed = processDataFrameInChunks(df=data_filtered, processor=processor)
        print("Exception Counts:", dict(exception_counters))
        print("--- %.2f seconds ---" % (time.time() - start_time))
        current_time = str(datetime.now().strftime("%Y-%m-%d %H-%M"))
        data_processed.to_csv(f'../output/productpage_classification_content_embeddings_based_productPhraseEmbeddings_dataset1_{current_time}.csv', index=False)

        plot_exception_histogram(dict(exception_counters))
