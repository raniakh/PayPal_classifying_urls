import os
from urllib.parse import urlparse, urlunparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import re
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

####
# DATA FRAME PREP FILE
####

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}


def makeLowerCase(column_name='sublinks'):
    print('## NOW RUNNING makeLowerCase')
    global sublinks

    def lowerCase(txt):
        return txt.lower()

    sublinks[column_name] = sublinks[column_name].apply(lowerCase)


def removeWWW(column_name='sublinks'):
    print('## NOW RUNNING removeWWW')
    global sublinks
    sublinks[column_name] = sublinks[column_name].str.replace('www.', '', regex=False)


def standardizeUrlWrapper(column_name='sublinks'):
    print('## NOW RUNNING standardizeUrlWrapper')
    global sublinks

    def standardize_url(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        path = parsed_url.path
        if path == '/' or path == '':
            return f'{scheme}://{domain}'
        return url

    sublinks[column_name] = sublinks[column_name].apply(standardize_url)


def extractURLcomponents(inputcol='sublinks'):
    print('## NOW RUNNING extractURLcomponents')
    global sublinks
    sublinks['domain'] = sublinks[inputcol].apply(lambda x: urlparse(x).netloc)
    sublinks['path'] = sublinks[inputcol].apply(lambda x: urlparse(x).path)
    sublinks['fragment'] = sublinks[inputcol].apply(lambda x: urlparse(x).fragment)


def createStopWordsSet():
    print('## NOW RUNNING createStopWordsSet')
    stop_words = set(stopwords.words('english'))
    url_top_words = ["pdf", "html"]
    stop_words.update(url_top_words)
    # Note: the word "about" was removed from nltk stop words file.
    return stop_words


def remove_specialCharsAndDigits(column_name):
    def clean_url(url):
        pattern = r'[\d_\.\-\:\?\$\%\/\=\+]'
        return re.sub(pattern, ' ', url)

    sublinks[column_name] = sublinks[column_name].apply(clean_url)


def tokenizeTxt(txt):
    tokens = word_tokenize(txt)
    return ' '.join(tokens)


def removeSpecialCharsAndStopWordsWrapper(column_names=('path', 'fragment')):
    print('## NOW RUNNING removeSpecialCharsAndStopWordsWrapper')

    def removeStopWords(txt):
        words = txt.split()
        filtered_words = [word for word in words if word not in stop_words_g and len(word) > 1]
        return ' '.join(filtered_words)

    for colname in column_names:
        remove_specialCharsAndDigits(colname)
        sublinks[colname] = sublinks[colname].apply(tokenizeTxt)
        sublinks[colname] = sublinks[colname].apply(removeStopWords)


def handleMissingValues():
    print('## NOW RUNNING handleMissingValues')
    global sublinks
    sublinks.fillna("None", inplace=True)
    for col in sublinks.columns[1:]:
        sublinks[col] = sublinks[col].replace("", "None")


def removeQuery():
    print('## NOW RUNNING removeQuery')
    global sublinks

    def remove_query(url):
        parsed_url = urlparse(url)
        url_without_query = urlunparse(parsed_url._replace(query=''))
        return url_without_query

    sublinks['sublinks'] = sublinks['sublinks'].apply(remove_query)


def removeDuplicates():
    print('## NOW RUNNING removeDuplicates')
    global sublinks

    sublinks = sublinks.drop_duplicates(subset='sublinks', keep='first').copy()


def removeFiles():
    print('## NOW RUNNING removeFiles')
    global sublinks
    file_extensions = ('.jpg', '.jpeg', '.png', '.pdf', '.gif', '.bmp', '.tiff',
                       '.jpg/', '.jpeg/', '.png/', '.pdf/', '.gif/', '.bmp/', '.tiff/')
    sublinks = sublinks[~sublinks['sublinks'].str.endswith(file_extensions)]


def lemmatizeTxt():
    print('## NOW RUNNING lemmatizeTxt')
    global sublinks
    wnl = WordNetLemmatizer()

    def preprocess_txt(txt):
        if not txt:
            return ""
        tokens = word_tokenize(txt)
        tokens = [wnl.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    for col in sublinks.columns[3:]:
        sublinks[col] = sublinks[col].apply(preprocess_txt)


def fetch_sample_text_for_lang_detect(url, max_length=500):
    try:
        response = requests.get(url=url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(['script', 'style', 'meta', 'link']):
            script_or_style.decompose()
        for hidden in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
            hidden.extract()
        visible_text = soup.get_text(separator=' ', strip=True)
        sample_text = ' '.join(visible_text.split())[:max_length]
        if not sample_text:
            print(f'No visible text found on the page, {url}')
            return ' '
        return sample_text

    except RequestException as e:
        print(f"Error fetching the webpage: {e}, url={url}")
        return ''
    except Exception as e:
        print(f"Error processing the webpage content: {e}, url={url}")


def removeNonEnglishWebsites():
    global sublinks
    domains = dict()
    print('## NOW RUNNING removeNonEnglishWebsites')

    def detect_language(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        clean_url = f'{scheme}://{domain}'
        try:
            if domain in domains.keys():
                return domains.get(domain)
            head_response = requests.head(clean_url, timeout=15)
            lang = head_response.headers.get('content-language')
            if lang:
                domains.update({domain: lang[:2]})
                return lang[:2]

            sample_text = fetch_sample_text_for_lang_detect(clean_url)
            if not sample_text:
                domains.update({domain: 'unknown'})
                return 'unknown'
            else:
                lang = detect(sample_text)
                domains.update({domain: lang[:2]})
                return lang
        except LangDetectException as e:
            print(f"Error detecting the language: {e}, {url}")
            return 'unknown'

    sublinks['language'] = sublinks['sublinks'].apply(detect_language)
    sublinks = sublinks[sublinks['language'] == 'en']


def urlLength():
    global sublinks
    sublinks['length'] = sublinks['sublinks'].apply(len)


def prepareDataFrame():
    global sublinks, stop_words_g
    stop_words_g = createStopWordsSet()
    makeLowerCase()
    removeWWW()
    removeQuery()
    standardizeUrlWrapper()
    removeDuplicates()
    removeFiles()
    removeNonEnglishWebsites()
    urlLength()
    extractURLcomponents()
    handleMissingValues()
    removeSpecialCharsAndStopWordsWrapper()
    handleMissingValues()


def preprocessDataFrame():
    lemmatizeTxt()
    current_time = str(datetime.now().strftime("%Y-%m-%d %H-%M"))
    print('## SAVING RESULTS..')
    sublinks.to_parquet(f'data/parquet_output/sublinks_depth7_{current_time}.parquet', index=False)
    sublinks.to_csv(f'data/parquet_output/sublinks_depth7_{current_time}.csv', index=False)


if __name__ == '__main__':
    start_time = time.time()

    parquet_folder_path = './data/parquet'
    parquet_files = [f for f in os.listdir(parquet_folder_path) if f.endswith('.parquet')]
    dataframes = [pd.read_parquet(os.path.join(parquet_folder_path, file)) for file in parquet_files]
    sublinks = pd.concat(dataframes, ignore_index=True)

    prepareDataFrame()
    preprocessDataFrame()
    print("--- %.2f seconds ---" % (time.time() - start_time))
