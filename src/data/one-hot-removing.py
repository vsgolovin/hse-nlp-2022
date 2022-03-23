import os
import numpy as np
import pandas as pd
import re
from string import punctuation
from nltk.stem import SnowballStemmer

def main():
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
    NUM_KEYWORDS = (3, 5, 10, 15, 25, 50, 100)
    for n in NUM_KEYWORDS:
        fname = os.path.join(DATA_DIR, f'top{n}.csv')
        df = pd.read_csv(fname, quotechar='"')
        with open(os.path.join(DATA_DIR, 'keywords.txt')) as f:
            keywords = np.array([w[:-1] for w in f])
        for _, row in df.iterrows():
            vector = np.array([int(x) for x in row['keywords']]) == 1
            vector[n:] = False
            row['keywords'] = keywords[vector].tolist()
        df.to_csv(fname, index=False)


if __name__ == '__main__':
    main()
