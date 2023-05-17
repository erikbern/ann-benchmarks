import sys

sys.path.append("install/lib-dolphinnpy")  # noqa
import numpy
from dolphinn import Dolphinn
from utils import findmean, isotropize

from ..base.module import BaseANN


class DolphinnPy(BaseANN):
    def __init__(self, num_probes):
        self.name = "Dolphinn(num_probes={} )".format(num_probes)
        self.num_probes = num_probes
        self.m = 1
        self._index = None

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = numpy.array(X, dtype=numpy.float32)
        d = X.shape[1]
        self.m = findmean(X, d, 10)
        X = isotropize(X, d, self.m)
        hypercube_dim = int(numpy.log2(len(X))) - 2
        self._index = Dolphinn(X, d, hypercube_dim)

    def query(self, v, n):
        q = numpy.array([v])
        q = isotropize(q, len(v), self.m)
        res = self._index.queries(q, n, self.num_probes)
        return res[0]
