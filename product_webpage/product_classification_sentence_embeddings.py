import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup, Comment
from sentence_transformers import SentenceTransformer
import time
from cachetools import cached, TTLCache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# This Python Script was used for experimenting on a lighter dataset.
# Divide the webpage content into sentences, and try to find the most similar sentence to the product phrase sentences.
# Conclusion:
# The method did not succeed in classifying the product webpages with a sufficient accuracy.



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

keywords = ['buy now', 'more payment options', 'buy with apple pay', 'pay now',
            'pay with paypal', 'buy with', 'sold out', 'out of stock', 'in stock',
            'product description', 'product specifications', 'product information',
            'customer reviews', 'product reviews', 'you may also like', 'related products',
            'more from this collection', 'other products', 'request a quote', 'view store information',
            'add to cart', 'order now', 'pay now', 'purchase', 'share product']

product_phrases = [
    "buy now and add to cart",
    "add to cart now and take advantage of our special discounts",
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

# cache with 1000 max items and 10 minutes time-to-live
cache = TTLCache(maxsize=1000, ttl=600)
nltk.download('punkt')
executor = ThreadPoolExecutor(max_workers=10)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def createStopWordsSet():
    print('## NOW RUNNING createStopWordsSet')
    stop_words = set(stopwords.words('english'))
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


def createPhraseEmbeddings():
    preprocessed_phrases = list(map(preprocessText, product_phrases))
    embed_phrases = model.encode(preprocessed_phrases)
    return embed_phrases


def fetchVisibleContent_Wrapper(df):
    print('## NOW RUNNING fetchVisibleContent_Wrapper')
    # data_filtered['Content'] = data_filtered['sublinks'].apply(fetchVisibleContent)
    df['Content'] = list(executor.map(fetchVisibleContent, df['sublinks']))


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


def preprocessText_Wrapper(df):
    print('## NOW RUNNING preprocessText_Wrapper')
    df.loc[:, 'Content_clean'] = df['Content'].apply(lambda x: preprocessTextSentences(x))
    df.loc[:, 'Content_clean'] = df['Content_clean'].replace('', 'None')
    df.drop(columns=['Content'], inplace=True)


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


def preprocessTextSentences(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    sentences = splitTextIntoSentences(text)
    sentences_preprocessed = []
    if stop_words_set is not None:
        for sentence in sentences:
            lst_text = sentence.split()
            lst_text = [word for word in lst_text if word not in
                        stop_words_set]

            lst_text = [ps.stem(word) for word in lst_text]
            lst_text = [lem.lemmatize(word) for word in lst_text]
            sentences_preprocessed.append(" ".join(lst_text))
    return sentences_preprocessed


def generateSentenceEmbedding_Wrapper(df):
    print('## NOW RUNNING generateSentenceEmbedding_Wrapper')
    df.loc[:, 'Sentence_embeds'] = df['Content_clean'].apply(lambda x: generateSentenceEmbedding(x))
    df.drop(columns=['Content_clean'], inplace=True)


def generateSentenceEmbedding(text):
    # # compare with phrase embeddings, classify accordingly
    # sentences = splitTextIntoSentences(text)
    # Generate embeddings for each chunk
    sentence_embeddings = model.encode(text)

    return sentence_embeddings


def splitTextIntoSentences(text, max_tokens=20):  # TODO TUNE max_tokens
    sentences = nltk.sent_tokenize(text)
    short_sentences = []

    for sentence in sentences:
        tokenized = model.tokenizer.encode(sentence, add_special_tokens=False)

        if len(tokenized) > max_tokens:
            for i in range(0, len(tokenized), max_tokens):
                chunk_tokens = tokenized[i:i + max_tokens]
                chunk_text = model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                short_sentences.append(chunk_text)
        else:
            # If the sentence is within the token limit, keep it as is
            short_sentences.append(sentence)
    return short_sentences


def classifyNewWebpages(sentences_matrix, similarity_threshold=0.55, min_threshold=2, max_threshold=3):
    # TODO CHANGE LOGIC compare similarity between embed sentences of new data and phrase embeds, count number of
    #  similar sentences above the similarity_threshold and classify according to num_threshold

    similarities = [cosine_similarity([embedding], product_phrase_embeds) for embedding in sentences_matrix]
    similarities_matrix = np.vstack(similarities)
    top_two_scores = []
    max_scores = np.max(similarities_matrix, axis=1)
    similar_sentences = np.where(max_scores >= similarity_threshold, 1, 0)
    if min_threshold <= np.sum(similar_sentences) <= max_threshold:
        return 1
    else:
        return 0


def classifyNewWebpages_Wrapper(newdata_df):
    newdata_df['Product Page'] = newdata_df['Sentence_embeds'].apply(classifyNewWebpages)


def plot_exception_histogram():
    print('## NOW RUNNING plot_exception_histogram')
    exception_names = list(exception_counters.keys())
    exception_counts = list(exception_counters.values())

    plt.bar(exception_names, exception_counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Exception Types')
    plt.ylabel('Frequency')
    plt.title('Histogram of Exceptions')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    exception_counters = defaultdict(int)

    stop_words_set = createStopWordsSet()
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()

    # Embed the phrases, compare embeddings of sentences with embeddings of phrases
    product_phrase_embeds = createPhraseEmbeddings()

    data_to_classify = pd.read_parquet('../data/parquet_output/sublinks_depth7_2024-10-28 16-15.parquet')
    fetchVisibleContent_Wrapper(df=data_to_classify)
    data_to_classify = data_to_classify[data_to_classify['Content'] != '']
    preprocessText_Wrapper(df=data_to_classify)
    generateSentenceEmbedding_Wrapper(df=data_to_classify)
    classifyNewWebpages_Wrapper(newdata_df=data_to_classify)
    data_to_classify.drop(columns=['Sentence_embeds'], inplace=True)
    current_time = str(datetime.now().strftime("%Y-%m-%d %H-%M"))
    data_to_classify.to_csv(f'../output/Testing_sentence_embeddings_threshold0.55_{current_time}.csv', index=False)
    plot_exception_histogram()
