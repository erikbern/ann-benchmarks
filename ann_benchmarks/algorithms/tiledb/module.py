from __future__ import absolute_import
import numpy
import os
import tiledb

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search import IVFFlatIndex
from tiledb.vector_search import FlatIndex
from tiledb.vector_search import VamanaIndex
from tiledb.cloud.dag import Mode
import numpy as np
import multiprocessing


from ..base.module import BaseANN

MAX_UINT64 = np.iinfo(np.dtype("uint64")).max

class TileDB(BaseANN):
    def __init__(self, metric, index_type, n_list = -1):
        self._index_type = index_type
        self._metric = metric
        self._n_list = n_list
        self._n_probe = -1
        self._opt_l = -1

    def query(self, v, n):
        if self._metric == 'angular':
            raise NotImplementedError()

        # query() returns a tuple of (distances, ids).
        ids = self.index.query(
            np.array([v]).astype(numpy.float32), 
            k=n, 
            nthreads=multiprocessing.cpu_count(), 
            nprobe=min(self._n_probe, self._n_list), 
            opt_l=self._opt_l
        )[1][0]
        # Fix for 'OverflowError: Python int too large to convert to C long'.
        ids[ids == MAX_UINT64] = 0
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
        self.res[self.res == MAX_UINT64] = 0

    def get_batch_results(self):
        return self.res

    def fit(self, X):
        array_uri = "/tmp/array"
        if os.path.isfile(array_uri):
            os.remove(array_uri)

        self.index = ingest(
            index_type=self._index_type,
            index_uri=array_uri,
            input_vectors=X,
            partitions=self._n_list,
        )
        if self._index_type == "IVF_FLAT":
            self.index = IVFFlatIndex(uri=array_uri)
        elif self._index_type == "FLAT":
            self.index = FlatIndex(uri=array_uri)
        elif self._index_type == "VAMANA":
            self.index = VamanaIndex(uri=array_uri)
        else:
            raise ValueError(f"Unsupported index {self._index_type}")

    def get_additional(self):
        return {}

class TileDBIVFFlat(TileDB):
    def __init__(self, metric, n_list):
        super().__init__(
            index_type="IVF_FLAT",
            metric=metric,
            n_list=n_list,
        )
    
    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def __str__(self):
        return 'TileDBIVFFlat(n_list=%d, n_probe=%d)' % (self._n_list, self._n_probe)

class TileDBFlat(TileDB):
    def __init__(self, metric, _):
        super().__init__(
            index_type="FLAT",
            metric=metric
        )
    
    def __str__(self):
        return 'TileDBFlat()'

class TileDBVamana(TileDB):
    def __init__(self, metric, _):
        super().__init__(
            index_type="VAMANA",
            metric=metric
        )
    
    def set_query_arguments(self, opt_l):
        self._opt_l = opt_l
    
    def __str__(self):
        return 'TileDBVamana(opt_l=%d)' % (self._opt_l)