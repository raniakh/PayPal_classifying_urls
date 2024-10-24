import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re

pattern1 = re.compile(r'/collections/.*/products/[^/]+$')  # /collections/.../products/ followed by product name
pattern5 = re.compile(r'/products/[^/]+')  # /products/ followed by product name
pattern2 = re.compile(r'/product/[^/]+')  # /product/ followed by product name
pattern3 = re.compile(r'/shop/[^/]+')  # /shop/ followed by product name
pattern4 = re.compile(r'/merch/p/[^/]+')  # /merch/p/ followed by product name


# extract words from url path
def wordAnalyzer(col='path'):
    print('## NOW RUNNING wordAnalyzer')
    global df
    col_name = col + '_tokens'
    df[col_name] = (df[col].astype(str)).apply(lambda x: x.split())
    all_tokens = [word for tokens in df[col_name] for word in tokens]
    word_counts = Counter(all_tokens)
    print("Most common words:", word_counts.most_common(10))

    word_counts_dict = dict(word_counts)
    most_common_words = word_counts.most_common(10)
    words, counts = zip(*most_common_words)
    plt.bar(words, counts)
    plt.title('Top 10 Most Common Words in URL Paths')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.show()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def basicStats():
    print('## NOW RUNNING basicStats')
    global df
    with pd.option_context('display.max_columns', 10):
        print('### df.describe ###')
        print(df.describe(include='all'))
    print('### df.shape ###')
    print(df.shape)
    print('### url length ###')
    print(df['length'].describe())
    print('### domain value counts ###')
    print(df['domain'].value_counts()[:10])


def tokenize_path(path):
    print('## NOW RUNNING tokenize_path')
    print(f'tokenize_path({path})')
    return path.split()


def wordsInPathByDomain():
    print('## NOW RUNNING wordsInPathByDomain')
    global df
    grouped_by_domain = df.groupby('domain')
    for domain, group in grouped_by_domain:
        all_words = [word for path in group['path'] for word in tokenize_path(path)]
        word_counts = Counter(all_words)
        print(f'Word Cloud for domain: {domain}')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for domain: {domain}")
        plt.savefig(fname=f'pics/parquet/bydomain/wordCloud domain {domain}.png', format='png')

        most_common_words = word_counts.most_common(10)
        words, counts = zip(*most_common_words)

        plt.figure(figsize=(8, 5))
        plt.bar(words, counts)
        plt.title(f"Top 10 Most Common Words in Path for domain: {domain}")
        plt.xticks(rotation=45)
        plt.ylabel('Frequency')
        plt.savefig(fname=f'pics/parquet/bydomain/barPlot domain {domain}.png', format='png')


def handleMissingValues():
    print('## NOW RUNNING handleMissingValues')
    global df
    df.fillna("None", inplace=True)
    for col in df.columns[1:]:
        df[col] = df[col].replace("", "None")


def patternFrequency():
    print('## NOW RUNNING patternFrequency')
    global df
    conditions = [
        df['sublinks'].str.contains(pattern1, regex=True),  # Pattern 1: /products/
        df['sublinks'].str.contains(pattern2, regex=True),  # Pattern 2: /product/
        df['sublinks'].str.contains(pattern3, regex=True),  # Pattern 3: /shop/
        df['sublinks'].str.contains(pattern4, regex=True),  # Pattern 4: /merch/p/
        df['sublinks'].str.contains(pattern5, regex=True)  # Pattern 5: /collections/.../products/
    ]
    choices = [
        'Pattern 1: /collections/.../products/',
        'Pattern 2: /product/',
        'Pattern 3: /shop/',
        'Pattern 4: /merch/p/',
        'Pattern 5: /products/'
    ]

    df['pattern'] = np.select(conditions, choices, default='Other')
    pattern_counts = df['pattern'].value_counts()
    plt.figure(figsize=(10, 6))

    sns.barplot(x=pattern_counts.index, y=pattern_counts.values, palette="viridis")

    plt.title('URL Pattern Frequencies', fontsize=16)
    plt.xlabel('URL Patterns', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    for index, value in enumerate(pattern_counts.values):
        plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.savefig(fname=f'./pics/parquet/pattern_frequencies.png', format='png')
    plt.show()


if __name__ == '__main__':
    # df = pd.read_csv('./data/sublinks_components_depth7.csv')
    df = pd.read_parquet('./data/parquet_output/sublinks_depth7_2024-10-14 16-43.parquet')
    handleMissingValues()
    basicStats()
    wordAnalyzer()
    wordsInPathByDomain()
    patternFrequency()
