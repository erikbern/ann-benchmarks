import sys
import os
import vamanapy as vp
import numpy as np
import struct
import time
from ann_benchmarks.algorithms.base import BaseANN


class Vamana(BaseANN):
    def __init__(self, metric, param):
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alpha = float(param["alpha"])
        print("Vamana: L_Build = " + str(self.l_build))
        print("Vamana: R = " + str(self.max_outdegree))
        print("Vamana: Alpha = " + str(self.alpha))
        self.params = vp.Parameters()
        self.params.set("L", self.l_build)
        self.params.set("R", self.max_outdegree)
        self.params.set("C", 750)
        self.params.set("alpha", self.alpha)
        self.params.set("saturate_graph", False)
        self.params.set("num_threads", 1)

    def fit(self, X):

        def bin_to_float(binary):
            return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

        print("Vamana: Starting Fit...")
        index_dir = 'indices'

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        data_path = os.path.join(index_dir, 'base.bin')
        self.name = 'Vamana-{}-{}-{}'.format(self.l_build,
                                             self.max_outdegree, self.alpha)
        save_path = os.path.join(index_dir, self.name)
        print('Vamana: Index Stored At: ' + save_path)
        shape = [np.float32(bin_to_float('{:032b}'.format(X.shape[0]))),
                 np.float32(bin_to_float('{:032b}'.format(X.shape[1])))]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print('Vamana: Creating Index')
            s = time.time()
            index = vp.SinglePrecisionIndex(vp.Metric.L2, data_path)
            index.build(self.params, [])
            t = time.time()
            print('Vamana: Index Build Time (sec) = ' + str(t - s))
            index.save(save_path)
        if os.path.exists(save_path):
            print('Vamana: Loading Index: ' + str(save_path))
            s = time.time()
            self.index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
            self.index.load(file_name = save_path)
            print("Vamana: Index Loaded")
            self.index.optimize_graph()
            print("Vamana: Graph Optimization Completed")
            t = time.time()
            print('Vamana: Index Load Time (sec) = ' + str(t - s))
        else:
            print("Vamana: Unexpected Index Build Time Error")

        print('Vamana: End of Fit')

    def set_query_arguments(self, l_search):
        print("Vamana: L_Search = " + str(l_search))
        self.l_search = l_search

    def query(self, v, n):
        return self.index.single_numpy_query(v, n, self.l_search)

    def batch_query(self, X, n):
        self.num_queries = X.shape[0]
        self.result = self.index.batch_numpy_query(X, n, self.num_queries, self.l_search)

    def get_batch_results(self):
        return self.result.reshape((self.num_queries, self.result.shape[0] // self.num_queries))
