import os
import numpy as np
import pandas as pd
import re
from string import punctuation
from nltk.stem import SnowballStemmer


def main():
    # open dataset
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
    df = pd.read_csv(os.path.join(DATA_DIR, 'dataset.csv'), quotechar='"')

    # process all abstracts
    for _, row in df.iterrows():
        row['abstract'] = process_abstract(row['abstract'])
    df.to_csv(os.path.join(DATA_DIR, 'full.csv'), index=False)

    # create subsets with less keywords
    NUM_KEYWORDS = (3, 5, 10, 15, 25, 50, 100)
    Y = np.array([list(map(int, mask)) for mask in df['keywords']],
                 dtype='bool')
    for n in NUM_KEYWORDS:
        mask = Y[:, :n].any(axis=1)
        print(f'{n} top keywords: {np.sum(mask)} abstracts')
        df_slice = df[mask]
        fname = 'processed_dataset_top'
        fname = os.path.join(DATA_DIR, f'top{n}.csv')
        df_slice.to_csv(fname, index=False)


def process_abstract(s: str) -> str:
    # make lowercase
    s = s.lower()  # done in stemmer

    # remove all backslash sequences
    s = re.sub(r'\\[\S]+', '', s)

    # remove citations
    s = re.sub(r'\[[0-9]+\]', '', s)

    # remove numbers
    s = re.sub(r'[0-9]+(?:\.[0-9]+)?', '', s)

    # remove punctuation
    for ch in punctuation:
        s = s.replace(ch, '')

    # change encoding to ascii and remove unicode characters
    s = s.encode(encoding='ascii', errors='ignore').decode()

    # remove tabs and multiple spaces
    # s = re.sub(r'\s+', ' ', s)  # done in stemmer

    # perform stemming
    # stemmer = SnowballStemmer('english', ignore_stopwords=False)
    # s = ' '.join(stemmer.stem(word) for word in re.split(r'\s+', s))

    return s


if __name__ == '__main__':
    main()
