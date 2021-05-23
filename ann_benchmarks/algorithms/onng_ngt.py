from __future__ import absolute_import
import sys
import os
import ngtpy
import numpy as np
import subprocess
import time
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR


class ONNG(BaseANN):
    def __init__(self, metric, object_type, epsilon, param):
        metrics = {'euclidean': '2', 'angular': 'C'}
        self._edge_size = int(param['edge'])
        self._outdegree = int(param['outdegree'])
        self._indegree = int(param['indegree'])
        self._metric = metrics[metric]
        self._object_type = object_type
        self._edge_size_for_search = int(param['search_edge']) if 'search_edge' in param.keys() else 0
        self._tree_disabled = (param['tree'] == False) if 'tree' in param.keys() else False
        self._refine_enabled = (param['refine'] == True) if 'refine' in param.keys() else False
        self._build_time_limit = 4
        self._epsilon = epsilon
        print('ONNG: edge_size=' + str(self._edge_size))
        print('ONNG: outdegree=' + str(self._outdegree))
        print('ONNG: indegree=' + str(self._indegree))
        print('ONNG: edge_size_for_search=' + str(self._edge_size_for_search))
        print('ONNG: epsilon=' + str(self._epsilon))
        print('ONNG: metric=' + metric)
        print('ONNG: object_type=' + object_type)

    def fit(self, X):
        print('ONNG: start indexing...')
        dim = len(X[0])
        print('ONNG: # of data=' + str(len(X)))
        print('ONNG: dimensionality=' + str(dim))
        index_dir = 'indexes'
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(
            index_dir,
            'ONNG-{}-{}-{}'.format(self._edge_size, self._outdegree,
                                   self._indegree))
        anngIndex = os.path.join(index_dir, 'ANNG-' + str(self._edge_size))
        print('ONNG: index=' + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print('ONNG: create ANNG')
            t = time.time()
            args = ['ngt', 'create', '-it', '-p8', '-b500', '-ga', '-of',
                    '-D' + self._metric, '-d' + str(dim),
                    '-E' + str(self._edge_size),
                    '-S' + str(self._edge_size_for_search),
                    '-e' + str(self._epsilon), '-P0', '-B30',
                    '-T' + str(self._build_time_limit), anngIndex]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=24, debug=False)
            print('ONNG: ANNG construction time(sec)=' + str(time.time() - t))
            t = time.time()
            if self._refine_enabled:
                idx.refine_anng(epsilon=self._epsilon, num_of_edges=self._edge_size,
                                num_of_explored_edges=self._edge_size_for_search)
            print('ONNG: RNNG construction time(sec)=' + str(time.time() - t))
            idx.save()
            idx.close()
        if not os.path.exists(index):
            print('ONNG: degree adjustment')
            t = time.time()
            args = ['ngt', 'reconstruct-graph', '-mS',
                    '-o ' + str(self._outdegree),
                    '-i ' + str(self._indegree), anngIndex, index]
            subprocess.call(args)
            print('ONNG: degree adjustment time(sec)=' + str(time.time() - t))
        if os.path.exists(index):
            print('ONNG: index already exists! ' + str(index))
            t = time.time()
            print(self._tree_disabled)
            self.index = ngtpy.Index(index, read_only=True, tree_disabled=self._tree_disabled)
            self.indexName = index
            print('ONNG: open time(sec)=' + str(time.time() - t))
        else:
            print('ONNG: something wrong.')
        print('ONNG: end of fit')

    def set_query_arguments(self, parameters):
        epsilon, edge_size = parameters
        print("ONNG: edge_size=" + str(edge_size))
        print("ONNG: epsilon=" + str(epsilon))
        self.name = 'ONNG-NGT(%s, %s, %s, %s, %1.3f)' % (
            self._edge_size, self._outdegree,
            self._indegree, edge_size, epsilon)
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, edge_size=edge_size)

    def query(self, v, n):
        return self.index.search(v, n, with_distance=False)

    def freeIndex(self):
        print('ONNG: free')
