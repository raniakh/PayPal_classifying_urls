import re
from urllib.parse import urlparse
import tldextract
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def remove_www(df, column_name='sublinks'):
    df[column_name] = df[column_name].str.replace('www.', '', regex=False)
    return df


def standardize_url_wrapper(df, column_name='sublinks'):
    def standardize_url(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        domain = parsed_url.netloc
        path = parsed_url.path
        if path == '/' or path == '':
            return f'{scheme}://{domain}'
        return url

    df[column_name] = df[column_name].apply(standardize_url)
    return df


# separators: ? query parameter , # anchor, / , & separate query parameters,
# =  assign values to parameters in the query
def calculate_lengthURL(df, column_name='sublinks'):
    df['url_length'] = df[column_name].apply(len)
    return df


def remove_specialCharsAndDigits(df, column_name='sublinks'):
    def clean_url(url):
        return re.sub(r'[\W_0-9$^]+', '', url)

    df['url_clean'] = df[column_name].apply(lambda x: x.split('https://')[-1])
    df['url_clean'] = df['url_clean'].apply(clean_url)
    return df


def extract_domain(df, column_name='sublinks'):
    df['domain'] = df[column_name].apply(lambda x: urlparse(x).netloc)
    df['domain_name'] = df['domain'].apply(lambda x: x.split('.')[0])
    return df


def extract_tld(df, column_name='domain'):
    df['tld'] = df[column_name].apply(lambda x: x.split('.')[-1])
    return df


def extract_second_level_domain_wrapper(df, column_name='domain'):
    def extract_second_level_domain(url):
        extracted = tldextract.extract(url)
        if extracted.suffix.count('.') > 0:
            parts = extracted.suffix.split('.')
            return parts[0] if len(parts) > 1 else None
        else:
            return None
    df['sld'] = df[column_name].apply(extract_second_level_domain)
    return df


if __name__ == '__main__':
    sublinks = pd.read_csv('data/sublinks.csv')
    sublinks = remove_www(sublinks)
    sublinks = standardize_url_wrapper(sublinks)
    sublinks = calculate_lengthURL(sublinks, 'sublinks')
    sublinks = remove_specialCharsAndDigits(sublinks)
    sublinks = extract_domain(sublinks)
    sublinks = extract_tld(sublinks)
    sublinks = extract_second_level_domain_wrapper(sublinks)
    sublinks.to_csv('output/clustering_stage2.csv', index=False)

