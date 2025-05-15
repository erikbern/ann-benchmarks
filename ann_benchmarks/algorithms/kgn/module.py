# import psutil
import os
import multiprocessing
from time import time

import gc
import numpy as np
import faiss
from faiss import Kmeans
from sklearn import preprocessing

import pykgn as kgn

from ..base.module import BaseANN




class EPSearcher:
    def __init__(self, data: np.ndarray, cur_ep: int) -> None:
        self.data = data
        self.cur_ep = cur_ep

    def search(self, query: np.ndarray) -> int:
        raise NotImplementedError

class EPSearcherKmeans_re(EPSearcher):
    def __init__(self, data: np.ndarray, cur_ep: int, max_deep: int, metric) -> None:
        super().__init__(data, cur_ep)
        self.centers = defaultdict(list)
        for i in range(1,max_deep+1):
            self.centers[i] = []
        final_centers = self.recursive_kmeans_centers(data, 2, max_deep)
        ncenters = 0
        cen = []
        for i in range(max_deep, 0, -1):
            ncenters += len(self.centers[i])
            for j in range(len(self.centers[i])):
                for k in range(len(self.centers[i][j])):
                    cen.append(self.centers[i][j][k])

        final = np.array(cen).reshape(ncenters, -1).astype('float32')

        raw_index = faiss.IndexFlatL2(data.shape[1])
        raw_index.add(data)
        _, self.RI = raw_index.search(final, 1)

    def recursive_kmeans_centers(self, data, num_clustters, max_deep):
        if max_deep == 1:
            kmeans = faiss.Kmeans(d=data.shape[1], k=num_clustters, verbose=False)
            kmeans.train(data)
            self.centers[max_deep].extend(kmeans.centroids.tolist())
            return kmeans.centroids
        kmeans = faiss.Kmeans(data.shape[1], num_clustters, seed=123, verbose=False)
        kmeans.train(data)
        _, labels = kmeans.index.search(data, 1)

        centers = kmeans.centroids

        self.centers[max_deep].extend(centers.tolist())
        result_centers = centers

        for i in range(num_clustters):
            subset_data = data[labels.reshape(-1) == i]
            subset_centers = self.recursive_kmeans_centers(subset_data, num_clustters, max_deep-1)
            result_centers = np.concatenate((result_centers,subset_centers))
        return result_centers

    def get_cent(self, )-> np.ndarray:
        return self.RI

def metric_mapping(metric):
    mapping_dict = {"angular": "IP", "euclidean": "L2"}
    metric_type = mapping_dict.get(metric)
    if metric_type is None:
        raise ValueError(f"The specified metric type '{metric}' is not recognized or supported by KGN.")
    return metric_type

class Kgn(BaseANN):
    def __init__(self, metric, dim, method_param):
        self.metric = metric_mapping(metric)
        self.name = 'kgn_(%s)' % (method_param)
        self.R = method_param['R']
        self.R2 = method_param['R2']
        self.level = method_param['level']
        self.dir = 'indices'
        self.path = f'{metric}_{dim}_{self.R}_{self.R2}_{self.level}.kgn'


    def build(self, X):
        Index = kgn.Index(nb=self.n, dim=self.d, base=X, topK=10, metric=self.metric, level=self.level, R=self.R, R2 = self.R2)
        full_path = os.path.join(self.dir, self.path)
        Index.build(full_path)

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        self.n = X.shape[0]
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path not in os.listdir(self.dir):
            full_path = os.path.join(self.dir, self.path)
            self.Index = kgn.Index(nb=self.n, dim=self.d, base=X, topK=10, metric=self.metric, level=self.level, R=self.R, R2 = self.R2)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                print(f"load Index in: '{full_path}'")               
                self.Index.load(full_path)
            else:
                print(f"build Index in: '{full_path}'")
                p = multiprocessing.Process(target=self.build, args=(X, ))
                p.start()
                p.join()
                gc.collect()
                self.Index.load(full_path)


    def set_query_arguments(self, reorder, prune, ef):
        if self.level == 2 and reorder == 1.5 :
            reorder = 1.2
        self.reorder = reorder
        self.prune = prune
        self.ef = ef
    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.Index.search(self.reorder, self.prune, self.ef, self.q)

    def get_prepared_query_results(self):
        return self.res

    def freeIndex(self):
        del self.Index