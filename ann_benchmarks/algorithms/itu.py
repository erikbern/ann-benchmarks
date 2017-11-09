from __future__ import absolute_import
import sys
sys.path.append('install/lib-annitu/build/wrappers/swig/')
import numpy
import locality_sensitive
from ann_benchmarks.algorithms.base import BaseANN

class ITUFilteringDouble(BaseANN):
    def __init__(self, metric, alpha = None, beta = None, threshold = None, tau = None, kappa1 = None, kappa2 = None, m1 = None, m2 = None):
        self._loader = locality_sensitive.double_vector_loader()
        self._context = None
        self._strategy = None
        self._metric = metric
        self._alpha = alpha
        self._beta = beta
        self._threshold = threshold
        self._tau = tau
        self._kappa1 = kappa1
        self._kappa2 = kappa2
        self._m1 = m1
        self._m2 = m2
        self.name = ("ITUFilteringDouble(..., threshold = %f, ...)" % threshold)

    def fit(self, X):
        if self._metric == 'angular':
            X /= numpy.linalg.norm(X, axis=1).reshape(-1,  1)
        self._loader.add(X)
        self._context = locality_sensitive.double_vector_context(
            self._loader, self._alpha, self._beta)
        self._strategy = locality_sensitive.factories.make_double_filtering(
            self._context, self._threshold,
            locality_sensitive.filtering_configuration.from_values(
                self._kappa1, self._kappa2, self._tau, self._m1, self._m2))

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        return self._strategy.find(v, n, None)

    def use_threads(self):
        return False

class ITUHashing(BaseANN):
    def __init__(self, seed, c = 2.0, r = 2.0):
        self._loader = locality_sensitive.bit_vector_loader()
        self._context = None
        self._strategy = None
        self._c = c
        self._r = r
        self._seed = seed
        self.name = ("ITUHashing(c = %f, r = %f, seed = %u)" % (c, r, seed))

    def fit(self, X):
        locality_sensitive.set_seed(self._seed)
        for entry in X:
            locality_sensitive.hacks.add(self._loader, entry.tolist())
        self._context = locality_sensitive.bit_vector_context(
            self._loader, self._c, self._r)
        self._strategy = locality_sensitive.factories.make_hashing(
            self._context)

    def query(self, v, n):
        return locality_sensitive.hacks.find(self._strategy, n, v.tolist())

    def use_threads(self):
        return False
