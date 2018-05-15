from __future__ import absolute_import
import sys
import os
#sys.path.append("/ann_bench_run/yj/ngt");
#import ngt
#from ngt import base as ngt
import ngtpy
import numpy as np
import subprocess
import time
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

class PANNG(BaseANN):
    def __init__(self, metric, object_type, edge_size, pathadj_size, edge_size_for_search, epsilon):
        metrics = {'euclidean': 'L2', 'angular': 'Cosine'}
        print('PANNG: edge_size=' + str(edge_size))
        print('PANNG: pathadj_size=' + str(pathadj_size))
        print('PANNG: edge_size_for_search=' + str(edge_size_for_search))
        print('PANNG: epsilon=' + str(epsilon - 1.0))
        print('PANNG: metric=' + metric)
        print('PANNG: object_type=' + object_type)
        self.name = 'PANNG-NGT(%d, %d, %d, %1.2f)' % (edge_size, pathadj_size, edge_size_for_search, epsilon)
        print('PANNG: name=' + self.name)
        self._metric = metrics[metric]
        self._object_type = object_type
        self._pathadj_size = int(pathadj_size)
        self._edge_size = int(edge_size)
        self._edge_size_for_search = int(edge_size_for_search)
        self._epsilon = float(epsilon) - 1.0

    def fit(self, X):
        print('PANNG: start indexing...')
        t0 = time.time()
        dim = len(X[0])
        print('PANNG: # of data=' + str(len(X)))
        print('PANNG: Dimensionality=' + str(dim))
        index_dir = 'indexes'
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(index_dir, 'NGT-' + str(self._edge_size))
        print(index)
        if os.path.exists(index):
            print('PANNG: index already exists! ' + str(index))
            self.index = ngtpy.Index(index)
            opentime = time.time() - t0
            print('PANNG: open time(sec)=' + str(opentime))
        else:
            ngtpy.create(path=index, dimension=dim, edge_size_for_creation=self._edge_size, distance_type=self._metric, 
                         object_type=self._object_type)
            idx = ngtpy.Index(path=index)
            idx.batch_insert(X, num_threads=24, debug=False)
            idx.save(index)
            idx.close()
            if self._pathadj_size > 0 :
                print('PANNG: path adjustment')
                args = ['ngt', 'prune', '-s ' + str(self._pathadj_size), index]
                subprocess.call(args)
            self.index = ngtpy.Index(path=index)
            indexingtime = time.time() - t0
            print('PANNG: indexing, adjustment and saving time(sec)=' + str(indexingtime))

    def query(self, v, n):
        results = self.index.search(v, n, self._epsilon, self._edge_size_for_search)
        return results

    def freeIndex(self):
        print('PANNG: free')
