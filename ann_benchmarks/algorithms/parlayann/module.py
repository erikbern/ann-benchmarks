from __future__ import absolute_import
import psutil
import os
import struct
import time
import numpy as np
import wrapper as pann

from ..base.module import BaseANN

class ParlayANN(BaseANN):
    def __init__(self, metric, index_params):
        self.name = "parlayann_(" + str(index_params) + ")"
        self._index_params = index_params
        self._metric = self.translate_dist_fn(metric)

        self.R = int(index_params.get("R", 50))
        self.L = int(index_params.get("L", 100))
        self.alpha = float(index_params.get("alpha", 1.15))
        self.two_pass = bool(index_params.get("two_pass", False))

    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return 'Euclidian'
        elif metric == 'ip':
            return 'mips'
        elif metric == 'angular':
            return 'mips'
        else:
            raise Exception('Invalid metric')
        
    def translate_dtype(self, dtype:str):
        if dtype == 'float32':
            return 'float'
        else:
            return dtype

    def fit(self, X):
        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        print("Vamana: Starting Fit...")
        index_dir = "indices"

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        data_path = os.path.join(index_dir, "base.bin")
        save_path = os.path.join(index_dir, self.name)
        print("parlayann: Index Stored At: " + save_path)
        nb, dims = X.shape
        shape = [
            np.float32(bin_to_float("{:032b}".format(nb))),
            np.float32(bin_to_float("{:032b}".format(dims))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("parlayann: Creating Index")
            start = time.time()
            self.params = pann.build_vamana_index(self._metric, "float", data_path, save_path,
                                                  self.R, self.L, self.alpha, self.two_pass)
            end = time.time()
            print("Indexing time: ", end - start)
            print(f"Wrote index to {save_path}")
        self.index = pann.load_index(self._metric, "float", data_path, save_path)
        print("Index loaded")

    def query(self, X, k):
        return self.index.single_search(X, k, self.Q, True, self.limit)

    def batch_query(self, X, k):
        print("running batch")
        nq, dims = X.shape
        self.res, self.distances = self.index.batch_search(X, k, self.Q, True, self.limit)
        return self.res

    def set_query_arguments(self, query_args):
        self.name = "parlayann_(" + str(self._index_params) + "," + str(query_args) + ")"
        print(query_args)
        self.limit = 1000 if query_args.get("limit") is None else query_args.get("limit")
        self.Q = 10 if query_args.get("Q") is None else query_args.get("Q")
