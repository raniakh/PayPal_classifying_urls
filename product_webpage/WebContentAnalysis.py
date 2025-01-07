from bs4 import BeautifulSoup, Comment
from collections import defaultdict
from cachetools import cached, TTLCache
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import time
import requests

cache = TTLCache(maxsize=1000, ttl=600)
nltk.download('punkt')
executor = ThreadPoolExecutor(max_workers=10)

target_phrases = [
    'buy now', 'more payment options', 'buy with apple pay', 'pay now',
    'pay with paypal', 'buy with',
    'sold out', 'out of stock', 'in stock',
    'product description', 'product specifications', 'product information',
    'customer reviews', 'product reviews',
    'you may also like', 'related products', 'more from this collection', 'other products',
    'request a quote', 'view store information', 'add to cart'
]


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


def plotFrequencies(percents, title, n=10):
    sorted_items = sorted(percents.items(), key=lambda x: x[1], reverse=True)[:n]
    labels, values = zip(*sorted_items)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(values)), values,
             tick_label=[' '.join(label) if isinstance(label, tuple) else label for label in labels])
    plt.xlabel('Percentage')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(fname=f'../pics/{title}.png', format='png')
    plt.show()


def count_phrases_in_text(text, phrases):
    """
    Return a dictionary {phrase: occurrence_count} for each phrase in the text.
    """
    text_lower = text.lower()
    phrase_counts = {}
    for phrase in phrases:
        phrase_counts[phrase] = text_lower.count(phrase.lower())
    return phrase_counts


if __name__ == '__main__':
    start_time = time.time()
    exception_counters = defaultdict(int)

    classified_data = pd.read_csv(
        '../stage2fileAsinput/productpage_classification_based_regex_dataset1_2024-10-16 18-25.csv')
    product_data = classified_data[classified_data['Product Page'] == 1].copy()

    fetchVisibleContent_Wrapper(df=product_data)
    product_data = product_data[product_data['Content'] != '']

    product_data['phrase_counts'] = product_data['Content'].apply(lambda x: count_phrases_in_text(x, target_phrases))

    # "flag dictionaries" indicating presence (1) or absence (0)
    product_data['phrase_flags'] = product_data['phrase_counts'].apply(
        lambda count_dict: {phrase: (1 if count_dict[phrase] > 0 else 0) for phrase in target_phrases}
    )

    # Aggregate the flags across all websites (sum them up)
    aggregated_flags = {phrase: 0 for phrase in target_phrases}
    for flags_dict in product_data['phrase_flags']:
        for phrase, flag in flags_dict.items():
            aggregated_flags[phrase] += flag

    num_websites = len(product_data)  # total number of websites
    phrase_percentages = {
        phrase: 100.0 * aggregated_flags[phrase] / num_websites
        for phrase in target_phrases
    }

    df_perc = pd.DataFrame(list(phrase_percentages.items()), columns=['phrase', 'percentage'])
    df_perc = df_perc.sort_values(by='percentage', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df_perc['phrase'], df_perc['percentage'], color='skyblue')
    plt.xlabel('Percentage of Websites', fontsize=12)
    plt.title('Percentage of Websites Containing Each Phrase', fontsize=14)
    plt.gca().invert_yaxis()  # highest percentage at the top
    plt.tight_layout()
    plt.show()
