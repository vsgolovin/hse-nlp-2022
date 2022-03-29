from os import path, getcwd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from dataset_loader import AbstractsDataset
from metrics import calculate_metrics, calculate_metrics_mean, Metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


DATA_DIR = path.join(getcwd(), 'data', 'processed') #'/home/margarita/HSE/ML/nlpproj/data/processed'
VOCAB_SIZE = 8192  # `None` to use all words


def main():
    # load dataset
    ax = []
    for N in [3, 5, 10, 15, 25]:
        DATASET = f'top{N}'
        fname = path.join(DATA_DIR, DATASET + '.csv')
        df = pd.read_csv(fname, quotechar='"', dtype=str)
        vectorizer = TfidfVectorizer(encoding='ascii', max_features=VOCAB_SIZE)
        X = vectorizer.fit_transform(df['abstract'])
        y = np.array([list(map(int, mask)) for mask in df['keywords']],
                     dtype='uint8')

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        # train and test model
        acc_test, prec_test, rec_test = [], [], []
        max_neigh = 90
        for neigh in range(5, max_neigh, 5):
            knn = KNeighborsClassifier(n_neighbors=neigh, weights="distance", n_jobs=-1)
            knn.fit(X_train, y_train)
            y_test_pred = knn.predict(X_test)
            a_test, p_test, r_test = calculate_metrics_mean(y_test_pred, y_test)
            acc_test.append(a_test)
            prec_test.append(p_test)
            rec_test.append(r_test)

        _, axn = plt.subplots()
        ax.append(axn)
        ax[-1] = plot_results(axn, max_neigh, acc_test, prec_test, rec_test, N)

    plt.show()


def plot_results(ax, max_neigh, acc, prec, rec, N):
    neigh = range(5, max_neigh, 5)
    ax.plot(neigh, acc, label="accuracy")
    ax.plot(neigh, prec, label='precision')
    ax.plot(neigh, rec, label="recall")
    ax.set_xlabel('Number of neighbors')
    ax.grid(linestyle=':')
    ax.title.set_text(f'Top {N} words')
    ax.legend()
    #ax.figure.savefig(f'top{N}_metrics.png')

if __name__ == '__main__':
    main()
