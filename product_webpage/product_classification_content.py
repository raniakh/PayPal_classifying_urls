import re
from datetime import datetime

import pandas as pd
import aiohttp
import asyncio
from urllib.parse import urlparse
import time
import matplotlib.pyplot as plt
from cachetools import cached, TTLCache
from collections import defaultdict
from joblib import Parallel, delayed
from multiprocessing import Manager
from bs4 import BeautifulSoup

# cache with 1000 max items and 10 minutes time-to-live
cache = TTLCache(maxsize=1000, ttl=600)

pattern = re.compile(
    r'/cart/*|/checkout/*$|'
    r'/login/*$|/signup/*|/log-in/*|/account/*|/my-account/*$|/register/*'
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


keywords = ['buy now', 'more payment options', 'buy with apple pay', 'pay now',
            'pay with paypal', 'buy with',
            'sold out', 'out of stock', 'in stock',
            'product description', 'product specifications', 'product information',
            'customer reviews', 'product reviews',
            'you may also like', 'related products', 'more from this collection', 'other products',
            'request a quote', 'view store information', 'add to cart']

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}


class URLProcessor:
    def __init__(self, exception_counters):
        self.exception_counters = exception_counters  # dictionary { exception-name: count}

    def matchRegex(self, chunk):
        chunk['regex_match'] = chunk['sublinks'].str.contains(pattern)
        chunk.loc[chunk['regex_match'], 'Product Page'] = -1
        return chunk.drop(columns=['regex_match'])

    def detectHomepage(self, chunk):
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
                    print(f"Failed to fetch {url}, statuse Code: {response.status}")
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

    # Extract webpage content and classify based on bag of words
    async def extractAndClassify_async(self, chunk):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetchContent(row['sublinks'], session) for index, row in chunk.iterrows()]
            contents = await asyncio.gather(*tasks)

            for i, content in enumerate(contents):
                if content:
                    visible_text = self.extractVisibleText(content)
                    keyword_matches = [key_word for key_word in keywords if key_word in visible_text.lower()]
                    if len(keyword_matches) >= 2:
                        chunk.iloc[i, chunk.columns.get_loc('Product Page')] = 1

        return chunk

    # Wrapper to run async function in sync code
    def extractAndClassifyWrapper(self, chunk):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.extractAndClassify_async(chunk))

    def processChunk(self, chunk):
        chunk = self.matchRegex(chunk)
        chunk = self.detectHomepage(chunk)

        filtered_chunk = chunk[chunk['Product Page'] == 0]
        if not filtered_chunk.empty:
            filtered_chunk = self.extractAndClassifyWrapper(filtered_chunk)

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
    chunks = list(chunkDataframe(df, chunk_size))
    results = Parallel(n_jobs=n_jobs)(delayed(processor.processChunk)(chunk) for chunk in chunks)

    return pd.concat(results)


def plot_exception_histogram(exception_counters_):
    exception_names = list(exception_counters_.keys())
    exception_counts = list(exception_counters_.values())

    plt.bar(exception_names, exception_counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Exception Types')
    plt.ylabel('Frequency')
    plt.title('Histogram of Exceptions')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()

    data_to_classify = pd.read_parquet('../data/parquet_output/sublinks_depth7_2024-10-28 16-15.parquet')
    data_to_classify['Product Page'] = 0
    # data_to_classify = pd.read_parquet('../output/product_classification_regex_time_2024-11-07 11-19_inputfile_sublinks_depth7_2024-10-28 16-15.parquet')
    # product_page_df = data_to_classify[data_to_classify['Product Page'] == 1].copy()
    # data_to_classify = data_to_classify[data_to_classify['Product Page'] != 1]

    current_time = str(datetime.now().strftime("%Y-%m-%d %H-%M"))
    with Manager() as manager:
        exception_counters = manager.dict(defaultdict(int))
        processor = URLProcessor(exception_counters)

        data_processed = processDataFrameInChunks(df=data_to_classify, processor=processor)
        print("Exception Counts:", dict(exception_counters))
        print("--- %.2f seconds ---" % (time.time() - start_time))

        data_processed.to_csv(
            f'../output/product_classification_content_time_{current_time}_inputfile_nostage2.csv',
            index=False)
        # data_concat = pd.concat([data_processed, product_page_df], ignore_index=True)
        # data_concat.to_csv(f'../output/product_classification_content_time_{current_time}_inputfile_stage2_concat2stages.csv', index=False)
        plot_exception_histogram(dict(exception_counters))
