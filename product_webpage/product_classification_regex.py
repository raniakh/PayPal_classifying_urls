import pandas as pd
import re
import numpy as np
from multiprocessing import Pool, cpu_count

pattern = re.compile(
    r'/products/[^/]+|'
    r'/product/[^/]+|'
    r'/shop/[^/]+|'
    r'/merch/p/[^/]+|'
    r'/collections/.*/products/[^/]+$'
)


def classify_chunk(df_chunk):
    global pattern
    df_chunk['Product Page'] = df_chunk['sublinks'].apply(
        lambda url: 1 if pattern.search(url) else 0
    )
    return df_chunk


def parallelize_dataframe(df, func, num_partitions=None):
    if num_partitions is None:
        num_partitions = cpu_count()

    df_split = np.array_split(df, num_partitions)

    with Pool(num_partitions) as pool:
        df = pd.concat(pool.map(func, df_split))

    return df


if __name__ == '__main__':
    data = pd.read_csv('../data/sublinks_components_depth7.csv')

    df_parallel = parallelize_dataframe(df=data, func=classify_chunk)
    df_parallel.to_csv('../output/productpage_classification_based_regex.csv', index=False)
