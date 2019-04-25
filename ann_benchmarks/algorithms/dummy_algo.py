from __future__ import absolute_import
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN


class DummyAlgoMt(BaseANN):
    def __init__(self, metric):
        self.name = 'DummyAlgoMultiThread'

    def fit(self, X):
        self.len = len(X) - 1

    def query(self, v, n):
        return np.random.randint(self.len, size=n)


class DummyAlgoSt(BaseANN):
    def __init__(self, metric):
        self.name = 'DummyAlgoSingleThread'

    def fit(self, X):
        self.len = len(X) - 1

    def query(self, v, n):
        return np.random.randint(self.len, size=n)
