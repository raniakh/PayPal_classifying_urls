import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


# extract words from url path
def wordAnalyzer(col='path'):
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
    print(f'tokenize_path({path})')
    return path.split()


def wordsInPathByDomain():
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
        plt.savefig(fname=f'pics/bydomain/wordCloud domain {domain}.png', format='png')

        most_common_words = word_counts.most_common(10)
        words, counts = zip(*most_common_words)

        plt.figure(figsize=(8, 5))
        plt.bar(words, counts)
        plt.title(f"Top 10 Most Common Words in Path for domain: {domain}")
        plt.xticks(rotation=45)
        plt.ylabel('Frequency')
        plt.savefig(fname=f'pics/bydomain/barPlot domain {domain}.png', format='png')


def handleMissingValues():
    global df
    df.fillna("None", inplace=True)
    for col in df.columns[1:]:
        df[col] = df[col].replace("", "None")


if __name__ == '__main__':
    df = pd.read_csv('../data/sublinks_components_depth7.csv')
    handleMissingValues()
    basicStats()
    wordAnalyzer()
    wordsInPathByDomain()
