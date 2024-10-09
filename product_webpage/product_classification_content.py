import re
import pandas as pd
import aiohttp
import asyncio
from urllib.parse import urlparse
import time
from cachetools import cached, TTLCache
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
# TODO at least two keywords present in the html
# TODO check if can see if attribute visible on page
keywords = ['buy now', 'more payment options', 'buy with apple pay', 'pay now',
            'pay with paypal', 'buy with',
            'sold out', 'out of stock', 'in stock',
            'product description', 'product specifications', 'product information',
            'customer reviews', 'product reviews',
            'you may also like', 'related products', 'more from this collection', 'other products',
            'request a quote', 'view store information']
# 'ratings', 'share product', 'reviews', 'sku', 'add to cart'
# 'credit card', 'debit card', 'debit or credit card',

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}


class URLProcessor:
    def __init__(self, timeout_counter):
        self.timeout_counter = timeout_counter

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
                    return await response.text()
                else:
                    print(f"Failed to fetch {url}, statuse Code: {response.status}")
                    return ''
        except asyncio.TimeoutError:
            print(f"Timeout fetching {url}")
            self.timeout_counter.value += 1
            return ''
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ''

    def extractVisibleText(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        for hidden in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
            hidden.extract()

        visible_text = soup.get_text(separator=' ')
        return ' '.join(visible_text.split())

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

    def getTimeOutCount(self):
        return self.timeout_counter.value


def chunkDataframe(df, chunk_size):
    for start in range(0, df.shape[0], chunk_size):
        yield df.iloc[start: start + chunk_size]


def processDataFrameInChunks(df, chunk_size=1000, n_jobs=-1, processor=None):
    chunks = list(chunkDataframe(df, chunk_size))
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(processor.processChunk)(chunk) for chunk in chunks)

    return pd.concat(results)


if __name__ == '__main__':
    start_time = time.time()

    data = pd.read_csv('../output/productpage_classification_based_regex.csv')
    data_filtered = data[data['Product Page'] == 0].copy()

    with Manager() as manager:
        timeout_counter = manager.Value('i', 0)
        processor = URLProcessor(timeout_counter)

        data_processed = processDataFrameInChunks(df=data_filtered, processor=processor)
        print(f"time out count: {processor.getTimeOutCount()}")
        print("--- %.2f seconds ---" % (time.time() - start_time))

        data_processed.to_csv('../output/productpage_classification_content_based.csv', index=False)
