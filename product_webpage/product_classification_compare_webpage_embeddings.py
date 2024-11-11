import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup, Comment
import time
from sentence_transformers import SentenceTransformer
from cachetools import cached, TTLCache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Code summary: take product webpages from stage 2, embed them and use them to compare with new embeddings


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
    df.loc[:, 'Content_clean'] = df['Content'].apply(lambda x: preprocessText(x))
    df.loc[:, 'Content_clean'] = df['Content_clean'].replace('', 'None')
    df.drop(columns=['Content'], inplace=True)


def splitTextIntoChunks(text, max_tokens=300):
    # Tokenize the entire text to get a list of token IDs
    tokenized = model.tokenizer.encode(text, add_special_tokens=False)

    # Split the tokenized text into chunks of max_tokens size
    chunks = []
    for i in range(0, len(tokenized), max_tokens):
        chunk_tokens = tokenized[i:i + max_tokens]
        # Decode the tokens back to text for each chunk
        chunk_text = model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

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
    print('## NOW RUNNING generateAverageEmbedding_Wrapper')
    df.loc[:, 'Avg_embed'] = df['Content_clean'].apply(lambda x: generateAverageEmbedding(x))
    df.drop(columns=['Content_clean'], inplace=True)


def softmax(matrix, axis=1):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=axis, keepdims=True))  # Subtract the max for numerical stability
    return exp_matrix / np.sum(exp_matrix, axis=axis, keepdims=True)


def classifyNewWebpages(product_df, newdata_df, threshold=0.95):
    print('## NOW RUNNING classifyNewWebpages')
    P_matrix = np.vstack(product_df['Avg_embed'].values)
    new_matrix = np.vstack(newdata_df['Avg_embed'].values)
    P_normalized = normalize(P_matrix, axis=1, norm='l2')
    new_normalized = normalize(new_matrix, axis=1, norm='l2')
    similarity_matrix = np.dot(P_normalized, new_normalized.T)
    max_scores = np.max(similarity_matrix, axis=0)
    classification = np.where(max_scores >= threshold, 1, 0)
    newdata_df['Product Page'] = classification
    newdata_df['Max Similarity Probability'] = max_scores


def sampleData(df):
    total_sample_size = 1000
    grouped_by_domain = df.groupby('domain')

    num_domains = product_data['domain'].nunique()
    min_sample_per_domain = max(1, total_sample_size // num_domains)

    sampled_data = grouped_by_domain.apply(lambda x: x.sample(n=min(min_sample_per_domain, len(x)), random_state=42))
    sampled_data = sampled_data.reset_index(drop=True)
    sampled_sublinks = set(sampled_data['sublinks'])

    if len(sampled_data) < total_sample_size:
        remaining_rows_needed = total_sample_size - len(sampled_data)
        additional_sample = product_data[~product_data['sublinks'].isin(sampled_sublinks)].sample(
            n=remaining_rows_needed, random_state=42)
        sampled_data = pd.concat([sampled_data, additional_sample]).reset_index(drop=True)

    return sampled_data


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
    exception_counters = defaultdict(int)

    classified_data = pd.read_csv(
        '../stage2fileAsinput/productpage_classification_based_regex_dataset1_2024-10-16 18-25.csv')
    product_data = classified_data[classified_data['Product Page'] == 1].copy()

    stop_words_set = createStopWordsSet()
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    # sample data by domain
    sampled_product = sampleData(df=product_data)
    product_data = sampled_product

    fetchVisibleContent_Wrapper(df=product_data)

    product_data = product_data[product_data['Content'] != '']

    preprocessText_Wrapper(df=product_data)

    generateAverageEmbedding_Wrapper(df=product_data)

    # data_to_classify = pd.read_parquet('../data/parquet_output/sublinks_depth7_2024-10-28 16-15.parquet')
    data_to_classify = pd.read_parquet(
        '../output/product_classification_regex_time_2024-11-07 11-19_inputfile_sublinks_depth7_2024-10-28 16-15.parquet')
    product_page_df = data_to_classify[data_to_classify['Product Page'] == 1].copy()
    data_to_classify = data_to_classify[data_to_classify['Product Page'] != 1]
    data_to_classify = data_to_classify.drop(columns=['Product Page'])

    fetchVisibleContent_Wrapper(df=data_to_classify)

    data_to_classify = data_to_classify[data_to_classify['Content'] != '']

    preprocessText_Wrapper(df=data_to_classify)
    generateAverageEmbedding_Wrapper(df=data_to_classify)

    classifyNewWebpages(product_df=product_data, newdata_df=data_to_classify)
    data_to_classify.drop(columns=['Avg_embed'], inplace=True)

    current_time = str(datetime.now().strftime("%Y-%m-%d %H-%M"))
    data_to_classify.to_csv(
        f'../output/product_classification_compare_webpage_embeddings_time_{current_time}_inputfile_stage2.csv',
        index=False)
    data_concat = pd.concat([data_to_classify, product_page_df], ignore_index=True)
    data_concat.to_csv(
        f'../output/product_classification_compare_webpage_embeddings_time_{current_time}_inputfile_stage2_concat2stages.csv',
        index=False)
    plot_exception_histogram(exception_counters)
