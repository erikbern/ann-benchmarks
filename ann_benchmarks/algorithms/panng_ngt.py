from __future__ import absolute_import
import sys
import os
import ngtpy
import numpy as np
import subprocess
import time
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR


class PANNG(BaseANN):
    def __init__(self, metric, object_type, param):
        metrics = {'euclidean': 'L2', 'angular': 'Cosine'}
        self._edge_size = int(param['edge'])
        self._pathadj_size = int(param['pathadj'])
        self._edge_size_for_search = int(param['searchedge'])
        self._metric = metrics[metric]
        self._object_type = object_type
        print('PANNG: edge_size=' + str(self._edge_size))
        print('PANNG: pathadj_size=' + str(self._pathadj_size))
        print('PANNG: edge_size_for_search=' + str(self._edge_size_for_search))
        print('PANNG: metric=' + metric)
        print('PANNG: object_type=' + object_type)

    def fit(self, X):
        print('PANNG: start indexing...')
        dim = len(X[0])
        print('PANNG: # of data=' + str(len(X)))
        print('PANNG: Dimensionality=' + str(dim))
        index_dir = 'indexes'
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(
            index_dir,
            'PANNG-' + str(self._edge_size) + '-' + str(self._pathadj_size))
        print(index)
        if os.path.exists(index):
            print('PANNG: index already exists! ' + str(index))
        else:
            t0 = time.time()
            ngtpy.create(path=index, dimension=dim,
                         edge_size_for_creation=self._edge_size,
                         distance_type=self._metric,
                         object_type=self._object_type)
            idx = ngtpy.Index(path=index)
            idx.batch_insert(X, num_threads=24, debug=False)
            idx.save()
            idx.close()
            if self._pathadj_size > 0:
                print('PANNG: path adjustment')
                args = ['ngt', 'prune', '-s ' + str(self._pathadj_size),
                        index]
                subprocess.call(args)
            indexingtime = time.time() - t0
            print('PANNG: indexing, adjustment and saving time(sec)={}'
                  .format(indexingtime))
        t0 = time.time()
        self.index = ngtpy.Index(path=index, read_only=True)
        opentime = time.time() - t0
        print('PANNG: open time(sec)=' + str(opentime))

    def set_query_arguments(self, epsilon):
        print("PANNG: epsilon=" + str(epsilon))
        self._epsilon = epsilon - 1.0
        self.name = 'PANNG-NGT(%d, %d, %d, %1.3f)' % (
            self._edge_size,
            self._pathadj_size,
            self._edge_size_for_search,
            self._epsilon + 1.0)

    def query(self, v, n):
        results = self.index.search(
            v, n, self._epsilon, self._edge_size_for_search,
            with_distance=False)
        return results

    def freeIndex(self):
        print('PANNG: free')
