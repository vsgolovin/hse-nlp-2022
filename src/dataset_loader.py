from typing import Union
import copy
import numpy as np
from scipy import sparse


class AbstractsDataset:
    def __init__(self, X: Union[np.ndarray, sparse.spmatrix],
                 y: Union[np.ndarray, sparse.spmatrix],
                 batch_size: int = 64):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.num_samples, self.num_features = X.shape
        self.batch_size = batch_size
        self.inds = np.arange(self.num_samples)

    def __len__(self):
        return self.num_samples

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def shuffle(self):
        self.inds = np.random.permutation(self.num_samples)

    def __iter__(self):
        start = 0
        while start < len(self):
            stop = min(start + self.batch_size, self.num_samples)
            inds = self.inds[start:stop]
            X = self.X[inds]
            if isinstance(X, sparse.spmatrix):
                X = X.toarray()
            y = self.y[inds]
            if isinstance(y, sparse.spmatrix):
                y = y.toarray()
            yield X, y
            start = stop
