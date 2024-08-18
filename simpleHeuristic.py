from urllib.parse import urlparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# TODO check the rate of "false" classes

def remove_duplicates_nulls(df):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


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


def determine_page_type(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    domain = parsed_url.netloc

    if path == '' or path == '/':
        return 'Homepage'

    if '/home' in path or '/index' in path:
        return 'Homepage'
    elif ('/store' in path or '/merch' in path or '/collections' in path
          or '/shop' in path or 'product-line' in path or 'products' in path):
        return 'Category Overview'
    elif 'product' in path or '/item' in path:
        return 'Product Page'
    elif '/contact' in path or '/contact-us' in path or '/contactus' in path or '/info' in path:
        return 'Help/Contact Us'
    elif '/about' in path or '/team' in path or '/aboutus' in path:
        return 'About'
    elif '/cart' in path or '/checkout' in path:
        return 'Cart'
    else:
        return 'Other'


if __name__ == '__main__':
    sublinks = pd.read_csv('data/sublinks.csv')
    sublinks = remove_www(sublinks)
    sublinks = standardize_url_wrapper(sublinks)
    sublinks['page_type'] = sublinks['sublinks'].apply(determine_page_type)
    sublinks.to_csv('init_page_type.csv', index=False)
