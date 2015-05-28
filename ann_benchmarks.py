import sys
sys.path.insert(0, '/Users/erikbern/tmp/lib/python/')
sys.path.insert(0, '/usr/local/share/flann/python')
sys.path.insert(0, '/Users/erikbern/tmp/panns')

import time
import numpy as np
import sklearn.neighbors
import annoy
import pyflann
import panns
import nearpy, nearpy.hashes, nearpy.distances

import sklearn.datasets

n_iter = 50
n_neighbors = 100

class LSHF(object):
    def __init__(self, n_estimators=10, n_candidates=50):
        self._lshf = sklearn.neighbors.LSHForest(n_candidates=50, n_neighbors=n_neighbors)

    def fit(self, X):
        self._lshf.fit(X)

    def query(self, v, n):
        return self._lshf.kneighbors(v,
                                     return_distance=False)[0]


class FLANN(object):
    def __init__(self, n_trees=10):
        self._flann = pyflann.FLANN(trees=n_trees)

    def fit(self, X):
        self._flann.build_index(X)

    def query(self, v, n):
        return self._flann.nn_index(v, n)[0][0]

class Annoy(object):
    def __init__(self, n_trees, n_candidates):
        self._n_trees = n_trees
        self._n_candidates = n_candidates

    def fit(self, X):
        self._annoy = annoy.AnnoyIndex(f=X.shape[1], metric='angular')
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), self._n_candidates)[:n]


class PANNS(object):
    def __init__(self, n_trees, n_candidates):
        self._n_trees = n_trees
        self._n_candidates = n_candidates

    def fit(self, X):
        self._panns = panns.PannsIndex(X.shape[1], metric='angular')
        for x in X:
            self._panns.add_vector(x)
        self._panns.build(self._n_trees)

    def query(self, v, n):
        return [x for x, y in self._panns.query(v, n)]


class NearPy(object):
    def __init__(self, n_bits):
        self._n_bits = n_bits

    def fit(self, X):
        nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp', self._n_bits)
        self._nearpy_engine = nearpy.Engine(X.shape[1], lshashes=[nearpy_rbp], distance=nearpy.distances.CosineDistance())

        for i, x in enumerate(X):
            self._nearpy_engine.store_vector(x.tolist(), i)

    def query(self, v, n):
        return [y for x, y, z in self._nearpy_engine.neighbours(v)]


class NearestNeighbors(object):
    def __init__(self):
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='cosine')

    def fit(self, X):
        self._nbrs.fit(X)

    def query(self, v, n):
        return self._nbrs.kneighbors(v, return_distance=False, n_neighbors=n)

algos = {
    'lshf': [LSHF(10, 50)],
    'flann': [FLANN(10)],
    'panns': [PANNS(10, 100)],
    'annoy': [Annoy(10, 100)],
    'nearpy': [NearPy(10)],
    'nn': [NearestNeighbors()],
}
        
X, labels_true = sklearn.datasets.make_blobs(n_samples=10000, n_features=100,
                                             centers=10, cluster_std=5,
                                             random_state=0)

for library in algos.keys():
    for algo in algos[library]:
        print library, algo, '...'
        t0 = time.time()
        algo.fit(X)
        print library, time.time() - t0

        i = 23
        v = X[i]
        print algo.query(v, 10)
        
