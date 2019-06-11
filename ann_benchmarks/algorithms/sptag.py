from __future__ import absolute_import
import SPTAG
from ann_benchmarks.algorithms.base import BaseANN


class Sptag(BaseANN):
    def __init__(self, metric, algo='BKT'):
        self._algo = str(algo)
        self._metric = str(metric)

    def fit(self, X):
        self._sptag = SPTAG.AnnIndex(self._algo, 'Float', X.shape[1])
	self._sptag.SetBuildParam("NumberOfThreads", '32')
	self._sptag.SetBuildParam("DistCalcMethod", self._metric)
        ret = self._sptag.Build(X.tobytes(), X.shape[0])

    def set_query_arguments(self, MaxCheck):
        self._sptag.SetSearchParam("MaxCheck", str(MaxCheck))

    def query(self, v, k):
        return self._sptag.Search(v.tobytes(), k)[0]

    def __str__(self):
        return 'Sptag(metric=%s, algo=%s)' % (self._metric,
                                                   self._algo)
