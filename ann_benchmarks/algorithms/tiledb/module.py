from __future__ import absolute_import
import numpy
import os
import tiledb

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search import IVFFlatIndex
from tiledb.cloud.dag import Mode
import numpy as np
import multiprocessing


from ..base.module import BaseANN

class TileDBIVFFlat(BaseANN):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric
        self.MAX_UINT64 = np.iinfo(np.dtype("uint64")).max

    def query(self, v, n):
        if self._metric == 'angular':
            raise NotImplementedError()

        # query() returns a tuple of (distances, ids).
        ids = self.index.query(
            np.array([v]).astype(numpy.float32), 
            k=n, 
            nthreads=multiprocessing.cpu_count(), 
            nprobe=min(self._n_probe,self._n_list), 
        )[1][0]
        # Fix for 'OverflowError: Python int too large to convert to C long'.
        ids[ids == self.MAX_UINT64] = 0
        return ids 

    def batch_query(self, X, n):
        if self._metric == 'angular':
            raise NotImplementedError()
        # query() returns a tuple of (distances, ids).
        self.res = self.index.query(
            X.astype(numpy.float32), 
            k=n, 
            nthreads=multiprocessing.cpu_count(), 
            nprobe=min(self._n_probe, self._n_list), 
        )[1]
        # Fix for 'OverflowError: Python int too large to convert to C long'.
        self.res[self.res == self.MAX_UINT64] = 0

    def get_batch_results(self):
        return self.res

    def fit(self, X):
        array_uri = "/tmp/array"
        if os.path.isfile(array_uri):
            os.remove(array_uri)

        self.index = ingest(
            index_type="IVF_FLAT",
            index_uri=array_uri,
            input_vectors=X,
            partitions=self._n_list,
        )
        self.index = IVFFlatIndex(uri=array_uri)

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def get_additional(self):
        return {}

    def __str__(self):
        return 'TileDBIVFFlat(n_list=%d, n_probe=%d)' % (self._n_list, self._n_probe)
