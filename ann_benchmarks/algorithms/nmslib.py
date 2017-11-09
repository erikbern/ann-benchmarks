from __future__ import absolute_import
import os
import nmslib
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN

class NmslibReuseIndex(BaseANN):
    @staticmethod
    def encode(d):
        return ["%s=%s" % (a, b) for (a, b) in d.iteritems()]
    def __init__(self, metric, method_name, index_param, save_index, query_param):
        self._nmslib_metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]
        self._method_name = method_name
        self._save_index = save_index
        self._index_param = NmslibReuseIndex.encode(index_param)
        self._query_param = NmslibReuseIndex.encode(query_param)
        self.name = 'Nmslib(method_name=%s, index_param=%s, query_param=%s)' % (self._method_name, self._index_param, self._query_param)
        self._index_name = os.path.join(INDEX_DIR, "nmslib_%s_%s_%s" % (self._method_name, metric, '_'.join(self._index_param)))

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
          os.makedirs(d)

    def fit(self, X):
        if self._method_name == 'vptree':
            # To avoid this issue:
            # terminate called after throwing an instance of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too big. Select the parameters so that <total # of records> is NOT less than <bucket size> * 1000
            # Aborted (core dumped)
            self._index_param.append('bucketSize=%d' % min(int(X.shape[0] * 0.0005), 1000))
                                        
        self._index = nmslib.init(self._nmslib_metric, [], self._method_name, nmslib.DataType.DENSE_VECTOR, nmslib.DistType.FLOAT)
    
        for i, x in enumerate(X):
            nmslib.addDataPoint(self._index, i, x.tolist())


        if os.path.exists(self._index_name):
            print "Loading index from file"
            nmslib.loadIndex(self._index, self._index_name)
        else:
            nmslib.createIndex(self._index, self._index_param)
            if self._save_index: 
              nmslib.saveIndex(self._index, self._index_name)

        nmslib.setQueryTimeParams(self._index, self._query_param)

    def query(self, v, n):
        return nmslib.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        nmslib.freeIndex(self._index)

class NmslibNewIndex(BaseANN):
    def __init__(self, metric, method_name, method_param):
        self._nmslib_metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]
        self._method_name = method_name
        self._method_param = NmslibReuseIndex.encode(method_param)
        self.name = 'Nmslib(method_name=%s, method_param=%s)' % (self._method_name, self._method_param)

    def fit(self, X):
        if self._method_name == 'vptree':
            # To avoid this issue:
            # terminate called after throwing an instance of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too big. Select the parameters so that <total # of records> is NOT less than <bucket size> * 1000
            # Aborted (core dumped)
            self._method_param.append('bucketSize=%d' % min(int(X.shape[0] * 0.0005), 1000))
                                        
        self._index = nmslib.init(self._nmslib_metric, [], self._method_name, nmslib.DataType.DENSE_VECTOR, nmslib.DistType.FLOAT)
    
        for i, x in enumerate(X):
            nmslib.addDataPoint(self._index, i, x.tolist())

        nmslib.createIndex(self._index, self._method_param)

    def query(self, v, n):
        return nmslib.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        nmslib.freeIndex(self._index)
