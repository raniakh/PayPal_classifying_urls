import pandas as pd
import re
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime

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
    data = pd.read_parquet('../data/parquet_output/sublinks_depth7_2024-10-28 16-15.parquet')

    df_parallel = parallelize_dataframe(df=data, func=classify_chunk)
    current_time = str(datetime.now().strftime("%Y-%m-%d %H-%M"))
    print('## SAVING RESULTS..')
    df_parallel.to_parquet(f'../output/product_classification_regex_time_{current_time}_inputfile_sublinks_depth7_2024-10-28 16-15.parquet', index=False)
    df_parallel.to_csv(f'../output/product_classification_regex_time_{current_time}_inputfile_sublinks_depth7_2024-10-28 16-15.csv', index=False)