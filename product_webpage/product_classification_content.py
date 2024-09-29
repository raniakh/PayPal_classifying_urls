# TODO take all pages that were classified NOT PRODUCT at the prev stage
#  regex mark homepage, cart, contactus,
#  login, contact, about us, privacy-policy,
#  search, account and more as NOT PRODUCT.
#  Extract content from the rest,
#  if webpage has words "add to card" "buy now".. then Product Page
import re
import pandas as pd
import aiohttp
import asyncio
from urllib.parse import urlparse
import time
from cachetools import cached, TTLCache
from joblib import Parallel, delayed

# cache with 1000 max items and 10 minutes time-to-live
cache = TTLCache(maxsize=1000, ttl=600)

pattern = re.compile(
    r'/cart/*$|/checkout/*$|'
    r'/login/*$|/signup/*$|/log-in/*$|/account/*$|/my-account/*$|'
    r'/return-policy/*$|/privacy-policy/*$|/terms-of-service/*$|/refund-policy/*$|/shipping-policy/*$|/terms-conditions'
    r'/*$|/terms-and-conditions/*$|'
    r'/reviews/*$|/all-reviews/*$|'
    r'/contact-us/*$|/contact/*$|/contactus/*$|/customer-service/*$|'
    r'/aboutus/*$|/about-us/*$|/about/*$|'
    r'/index.php/*$|/home/*$|'
    r'/blog/*$|/news/*$|/our-brands/*$|'
    r'/search/*$|/collections/*$|/merch/*$|'
    r'/collections/|/category/|/category/collections/'
)

keywords = ['buy now', 'add to cart', 'more payment options', 'buy with apple pay',
            'pay with paypal', 'buy with', 'sold out', 'out of stock', 'in stock',
            'credit card', 'debit card', 'debit or credit card',
            'description', 'specs', 'product description', 'product specifications',
            'product information',
            'reviews', 'customer reviews', 'ratings', 'share product',
            'you may also like', 'related product', 'more from this collection',
            'sku', 'request a quote']


def matchRegex(chunk):
    chunk['regex_match'] = chunk['sublinks'].str.contains(pattern)
    chunk.loc[chunk['regex_match'], 'Product Page'] = -1
    return chunk.drop(columns=['regext_match'])


def detectHomepage(chunk):
    def isHomepage(url):
        parsed_url = urlparse(url)
        return parsed_url.path in ['', '/', '/home', '/index.php']

    chunk['homepage_match'] = chunk['sublinks'].apply(isHomepage)
    chunk.loc[chunk['homepage_match'], 'Product Page'] = -1
    return chunk.drop(column=['homepage_match'])


@cached(cache)
async def fetchContent(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    return await response.text()
                return ''
    except Exception:
        return ''


# Extract webpage content and classify based on bag of words
async def extractAndClassify_async(chunk):
    tasks = [fetchContent(row['sublinks']) for index, row in chunk.iterrows()]
    contents = await asyncio.gather(*tasks)

    for i, content in enumerate(contents):
        if any(key_word in content.lower() for key_word in keywords):
            chunk.iloc[i, chunk.columns.get_loc('Product Page')] = 1

    return chunk


# Wrapper to run async function in sync code
def extractAndClassifyWrapper(chunk):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(extractAndClassify_async(chunk))


def chunkDataframe(df, chunk_size):
    for start in range(0, df.shape[0], chunk_size):
        yield df.iloc[start: start + chunk_size]


def processChunk(chunk):
    chunk = matchRegex(chunk)
    chunk = detectHomepage(chunk)

    filtered_chunk = chunk[chunk['Product Page'] == 0]
    if not filtered_chunk.empty:
        filtered_chunk = extractAndClassifyWrapper(filtered_chunk)

    chunk.update(filtered_chunk)
    return chunk


def processDataFrameInChunks(df, chunk_size=1000, n_jobs=-1):
    chunks = list(chunkDataframe(df, chunk_size))
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(processChunk)(chunk) for chunk in chunks)

    return pd.concat(results)


if __name__ == '__main__':
    start_time = time.time()
    data = pd.read_csv('../output/productpage_classification_based_regex.csv')
    data_filtered = data[data['Product Page'] == 0].copy()
    data_processed = processDataFrameInChunks(data_filtered)
    print("--- %.2f seconds ---" % (time.time() - start_time))
    data_processed.to_csv('./output/productpage_classification_content_based.csv', index=False)
