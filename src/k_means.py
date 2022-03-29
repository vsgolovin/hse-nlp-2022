from os import path, getcwd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from dataset_loader import AbstractsDataset
from metrics import calculate_metrics, calculate_metrics_mean, Metrics

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

global clust_max, step
clust_max = None
step = None


DATA_DIR = path.join(getcwd(), 'data', 'processed')
FIG_DIR = path.join(getcwd(), 'reports', 'k_means')
VOCAB_SIZE = 8192  # `None` to use all words
points = 10

def main():
    for N in [5, 10, 15]:

        DATASET = f'top{N}'

        # load dataset
        fname = path.join(DATA_DIR, DATASET + '.csv')
        df = pd.read_csv(fname, quotechar='"', dtype=str)
        vectorizer = TfidfVectorizer(encoding='ascii', max_features=VOCAB_SIZE)
        X = vectorizer.fit_transform(df['abstract'])
        print(X.shape)
        y = np.array([list(map(int, mask)) for mask in df['keywords']],
                     dtype='uint8')
        global clust_max, step
        clust_max = int(1.5 * len(np.unique(y, axis = 0)))
        step = (clust_max * 2 - N) // points

        ax = []
        acc_test, prec_test, rec_test = [], [], []

        for N_clust in range(N, clust_max, step):
            # train-validate-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)


            # train and test model
            kmeans = KMeans(n_clusters=N_clust, init='k-means++')
            y_train_pred = kmeans.fit_predict(X_train)
            y_test_pred = kmeans.predict(X_test)

            # learn object that finds the most popular array of keywords (one-hot vectors)
            # and decode predicted cluster numbers to the arrays of keywords
            cluster_decoder = cluster_dictionary()
            cluster_decoder.fit(y_train_pred, y_train)
            y_test_pred_vect = cluster_decoder.decode(y_test_pred)

            acc, prec, rec = calculate_metrics_mean(y_test_pred_vect, y_test)
            print(f'acc = {acc:.3f}, prec = {prec:.3f}, rec = {rec:.3f}')
            acc_test.append(acc)
            prec_test.append(prec)
            rec_test.append(rec)
        _, axn = plt.subplots()
        ax.append(axn)
        ax[-1] = plot_results(axn, acc_test, prec_test, rec_test, N)
    plt.show()


def plot_results(ax, acc, prec, rec, N):
    N_clust = range(N, clust_max, step)
    ax.plot(N_clust, acc, label="accuracy")
    ax.plot(N_clust, prec, label='precision')
    ax.plot(N_clust, rec, label="recall")
    ax.set_xlabel('Number of neighbors')
    ax.grid(linestyle=':')
    ax.title.set_text(f'Top {N} words')
    ax.legend()
    ax.figure.savefig(path.join(FIG_DIR, f'top{N}_metrics.png'))

class cluster_dictionary:
    def __init__(self):
        # self.n = n
        self.d = {}
        self.d_encoder = {}
        self.n = None

    def fit(self, y_pred, y_true):
        """
        y_pred - predicted numbers of clusters that vectors correspond to
        y_true - one-hot vectors (keywords)
        function learns dictionary to decode cluster numbers to the most
        popular one-hot-vectors in the clusters
        """
        self.n = y_true.shape[1]
        for pred, true in zip(y_pred, y_true):
            if pred not in self.d:
                self.d[pred] = [true]
            else:
                self.d[pred].append(true)
        for cluster in self.d:
            unique, count = np.unique(self.d[cluster], return_counts=True, axis=0)
            self.d_encoder[cluster] = unique[np.argmax(count)]

    def decode(self, y_pred):
        res = np.zeros((y_pred.shape[0], self.n))
        for i in range(len(y_pred)):
            res[i, :] = self.d_encoder[y_pred[i]]
        return res.astype('uint8')

def _test():
    cluster_decoder = cluster_dictionary()
    vectors = np.array([[0, 0], [1, 1], [1, 1], [1, 1], [1, 2], [1, 2], [-2, 1], [-2, 1]])
    labels = np.array([0, 1, 1, 1, 1, 1, 0, 0])
    cluster_decoder.fit(labels, vectors)
    print(cluster_decoder.d)
    print(cluster_decoder.d_encoder)

if __name__ == '__main__':
    main()

