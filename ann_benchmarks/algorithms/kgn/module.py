import psutil
import os
from time import time
from sklearn import preprocessing

import pykgn as kgn
import numpy as np
import faiss
from faiss import Kmeans

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
        self.R = method_param['R']
        self.L = method_param['L']
        self.index_type = method_param['index_type']
        self.optimize = method_param['optimize']
        self.batch = method_param['batch']
        self.kmeans_ep = method_param['kmeans_ep']
        self.kmeans_type = method_param['kmeans_type']
        self.level = method_param['level']
        self.name = 'kgn_(%s)' % (method_param)
        self.dir = 'indices'
        self.path = f'{metric}_{dim}_{self.index_type}_R_{self.R}_L_{self.L}.kgn'
        
    def fit(self, X):
        print(self.name, self.level, self.metric)
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path not in os.listdir(self.dir):
            print("build Index")
            p = kgn.Index(self.index_type, dim=self.d,
                            metric=self.metric, R=self.R, L=self.L) 
            g = p.build(X,20)
            g.save(os.path.join(self.dir, self.path))
            del p
            del g

        # find kmeans centers -- RI
        if(self.kmeans_type==0):
            RI = np.array([])
        elif(self.kmeans_type==2):
            t = time()
            kmeans_ep_searcher = EPSearcherKmeans_re(X, 0, self.kmeans_ep, self.metric)
            T = time() - t
            print("Time of bi_kmeans  = ", T, " k=", self.kmeans_ep)
            RI = kmeans_ep_searcher.get_cent()
        else:
            print("Error: no such kmeans algorithm in main_opt.py")
        print("kmeans_ep", self.kmeans_ep)
        g = kgn.Graph()
        g.load(os.path.join(self.dir, self.path))
        if self.level == 1:
            self.searcher = kgn.Searcher(g, X, self.metric, "SQ8U",20)
        elif self.level == 2:
            self.searcher = kgn.Searcher(g, X, self.metric, "SQ4U",20)
        print("Make Searcher")

        if self.optimize:
            if self.batch:
                if self.level <= 4:
                    self.searcher.optimize()
                else:
                    print(self.level, "no needs optimized")
                    pass
            else:
                if self.level <= 4:
                    self.searcher.optimize(1)
                else:
                    print(self.level, "no needs optimized")
                    pass
        print("Optimize Parameters")
        

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)
        self.ef = ef

    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = q
        self.n = n

    def run_prepared_query(self):
        if self.level <= 3:
            self.res = self.searcher.search(
                self.q, self.n)
        else:
            self.res = self.searcher.search(
                self.q, self.n)

    def get_prepared_query_results(self):
        return self.res

    def freeIndex(self):
        del self.searcher
