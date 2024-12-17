import requests
import pandas as pd


def check_shopify(url):
    try:
        # Append '/robots.txt' to the URL
        robots_url = url.rstrip('/') + '/robots.txt'
        response = requests.get(robots_url, timeout=5)

        # Check if the robots.txt file contains Shopify-specific markers
        if response.status_code == 200:
            if 'shopify' in response.text.lower():
                return 'Shopify'
            elif 'woocomerce' in response.text.lower():
                return 'Woocommerce'
            elif 'squarespace' in response.text.lower():
                return 'Squarespace'
            elif 'wix' in response.text.lower():
                return 'Wix'
            else:
                return 'None'
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")


if __name__ == '__main__':
    urls = pd.read_csv('../data/datahack_sample.csv')
    urls['Built with'] = urls['seed_url'].apply(check_shopify)
    urls.to_csv('../data/datahack_sample_built_with.csv', index=False)