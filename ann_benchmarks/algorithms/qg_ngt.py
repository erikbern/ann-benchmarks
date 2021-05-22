from __future__ import absolute_import
import sys
import os
import ngtpy
import numpy as np
import subprocess
import time
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

class QG(BaseANN):
    def __init__(self, metric, object_type, epsilon, param):
        metrics = {'euclidean': '2', 'angular': 'E'}
        self._edge_size = int(param['edge'])
        self._outdegree = int(param['outdegree'])
        self._indegree = int(param['indegree'])
        self._max_edge_size = int(param['max_edge']) if 'max_edge' in param.keys() else 128
        self._metric = metrics[metric]
        self._object_type = object_type
        self._edge_size_for_search = int(param['search_edge']) if 'search_edge' in param.keys() else -2
        self._tree_disabled = (param['tree'] == False) if 'tree' in param.keys() else False
        self._build_time_limit = 4
        self._epsilon = epsilon
        print('QG: edge_size=' + str(self._edge_size))
        print('QG: outdegree=' + str(self._outdegree))
        print('QG: indegree=' + str(self._indegree))
        print('QG: edge_size_for_search=' + str(self._edge_size_for_search))
        print('QG: epsilon=' + str(self._epsilon))
        print('QG: metric=' + metric)
        print('QG: object_type=' + object_type)

    def fit(self, X):
        print('QG: start indexing...')
        dim = len(X[0])
        print('QG: # of data=' + str(len(X)))
        print('QG: dimensionality=' + str(dim))
        index_dir = 'indexes'
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(
            index_dir,
            'ONNG-{}-{}-{}'.format(self._edge_size, self._outdegree,
                                   self._indegree))
        anngIndex = os.path.join(index_dir, 'ANNG-' + str(self._edge_size))
        print('QG: index=' + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print('QG: create ANNG')
            t = time.time()
            args = ['ngt', 'create', '-it', '-p8', '-b500', '-ga', '-of',
                    '-D' + self._metric, '-d' + str(dim),
                    '-E' + str(self._edge_size), '-S40',
                    '-e' + str(self._epsilon), '-P0', '-B30',
                    '-T' + str(self._build_time_limit), anngIndex]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=24, debug=False)
            idx.save()
            idx.close()
            print('QG: ANNG construction time(sec)=' + str(time.time() - t))
        if not os.path.exists(index):
            print('QG: degree adjustment')
            t = time.time()
            args = ['ngt', 'reconstruct-graph', '-mS',
                    '-E ' + str(self._outdegree),
                    '-o ' + str(self._outdegree),
                    '-i ' + str(self._indegree), anngIndex, index]
            subprocess.call(args)
            print('QG: degree adjustment time(sec)=' + str(time.time() - t))
        if not os.path.exists(index + '/qg'):
            print('QG: quantization')
            t = time.time()
            args = ['ngtqg', 'quantize', index]
            subprocess.call(args)
            print('QG: quantization time(sec)=' + str(time.time() - t))
        if os.path.exists(index):
            print('QG: index already exists! ' + str(index))
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print('QG: open time(sec)=' + str(time.time() - t))
        else:
            print('QG: something wrong.')
        print('QG: end of fit')

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        print("QG: result_expansion=" + str(result_expansion))
        print("QG: epsilon=" + str(epsilon))
        self.name = 'QG-NGT(%s, %s, %s, %s, %s, %1.3f)' % (
            self._edge_size, self._outdegree,
            self._indegree, self._max_edge_size,
            epsilon,
            result_expansion)
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def freeIndex(self):
        print('QG: free')
