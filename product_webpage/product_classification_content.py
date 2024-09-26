# TODO take all pages that were classified NOT PRODUCT at the prev stage
#  regex mark homepage, cart, contactus,
#  login, contact, about us, privacy-policy,
#  search, account and more as NOT PRODUCT.
#  Extract content from the rest,
#  if webpage has words "add to card" "buy now".. then Product Page
import pandas as pd
import re
import time
import numpy as np
from joblib import Parallel, delayed

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
    r'/search/*$|/collections/*$|merch/*$'
)

keywords = ['buy now', 'add to cart', 'product description']  # TODO add more keywords


def non_product_pages_regex():
    data_filtered['regex_match'] = data_filtered['sublinks'].str.contains(pattern)


def homepage_detect():
    pass


if __name__ == '__main__':
    start_time = time.time()
    data = pd.read_csv('./output/productpage_classification_based_regex.csv')
    data_filtered = data[data['Product Page'] == 0].copy()
    print("--- %.2f seconds ---" % (time.time() - start_time))
