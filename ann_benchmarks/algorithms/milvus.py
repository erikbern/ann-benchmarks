from __future__ import absolute_import
import milvus
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class Milvus(BaseANN):
    def __init__(self, metric, index_type, nlist):
        self._nlist = nlist
        self._nprobe = None
        self._metric = metric
        self._milvus = milvus.Milvus()
        self._milvus.connect(host='localhost', port='19530')
        self._table_name = 'test01'
        self._index_type = index_type

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        self._milvus.create_table({'table_name': self._table_name, 'dimension': X.shape[1]})
        vector_ids = [id for id in range(len(X))]
        self._milvus.insert(table_name=self._table_name, records=X.tolist(), ids=vector_ids)
        index_type = getattr(milvus.IndexType, self._index_type)  # a bit hacky but works
        self._milvus.create_index(self._table_name, {'index_type': index_type, 'nlist': self._nlist})

    def set_query_arguments(self, nprobe):
        if nprobe > self._nlist:
            print('warning! nprobe > nlist')
            nprobe = self._nlist
        self._nprobe = nprobe

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        v = v.tolist()
        status, results = self._milvus.search(table_name=self._table_name, query_records=[v], top_k=n, nprobe=self._nprobe)
        if not results:
            return []  # Seems to happen occasionally, not sure why
        result_ids = [result.id for result in results[0]]
        return result_ids

    def __str__(self):
        return 'Milvus(index_type=%s, nlist=%d, nprobe=%d)' % (self._index_type, self._nlist, self._nprobe)
