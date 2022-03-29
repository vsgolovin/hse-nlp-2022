from os import path, getcwd
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, naive_bayes
from metrics import calculate_metrics_mean


DATA_DIR = path.join(getcwd(), 'data', 'processed')
VOCAB_SIZE = 2048  # `None` to use all words
DATASET = 'top5'


def main():
    # load dataset
    fname = path.join(DATA_DIR, DATASET + '.csv')
    df = pd.read_csv(fname, quotechar='"', dtype=str)
    vectorizer = TfidfVectorizer(encoding='ascii', max_features=VOCAB_SIZE)
    X = vectorizer.fit_transform(df['abstract'])
    y = np.array([list(map(int, mask)) for mask in df['keywords']],
                 dtype='uint8')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True
    )
    NUM_LABELS = y.shape[1]

    # binary classifiers
    classifiers = (
        svm.LinearSVC,
        tree.DecisionTreeClassifier,
        naive_bayes.MultinomialNB
    )
    settings = (
        {},
        dict(max_depth=7, min_samples_leaf=5),
        {}
    )
    names = (
        'SVM',
        'Decision Tree',
        'Naive Bayes'
    )

    for bin_clf, kwargs, name in zip(classifiers, settings, names):
        print(f'{name}:')
        clf = MultiLabelClassifier(NUM_LABELS, bin_clf, kwargs)
        try:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
        except TypeError:
            clf.fit(X_train.toarray(), y_train)
            pred = clf.predict(X_test.toarray())
        metrics = calculate_metrics_mean(pred, y_test)
        print(metrics, '\n')


class MultiLabelClassifier:
    def __init__(self, num_labels: int, binary_classifier: Callable, kwargs):
        self.num_labels = num_labels
        self.classifiers = [binary_classifier(**kwargs)
                            for _ in range(num_labels)]

    def fit(self, X, y):
        for i, clf in enumerate(self.classifiers):
            clf.fit(X, y[:, i])

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_labels), dtype='int')
        for i, clf in enumerate(self.classifiers):
            predictions[:, i] = clf.predict(X)
        return predictions


if __name__ == '__main__':
    main()
