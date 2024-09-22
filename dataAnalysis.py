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
    df[col_name] = df[col].apply(lambda x: x.split())
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

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(word_counts_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def basicStats():
    global df
    print('### df.describe ###')
    print(df.describe(include='all'))
    print('### df.shape ###')
    print(df.shape)
    print('### url length ###')
    print(df['length'].describe())
    print('### domain value counts ###')
    print(df['domain'].value_counts()[:10])


if __name__ == '__main__':
    df = pd.read_csv('data/sublinks_components_depth7_withNonEnglish.csv')
    basicStats()
