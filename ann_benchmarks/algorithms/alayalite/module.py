import os
from dataclasses import dataclass, field
import numpy as np
from alayalite import Client
from alayalite import Index
from ..base.module import BaseANN


class AlayaLite(BaseANN):
    def __init__(self, metric, dim, method_param):
        self.index_save_dir = 'alaya_indices'
        self.client = Client(self.index_save_dir)
        self.index = None
        self.ef = None
        self.dim = dim
        self.metric = metric

        self.index_type = method_param['index_type']
        self.quantization_type = method_param['quantization_type']
        self.fit_threads = method_param['fit_threads']
        self.search_threads = method_param['search_threads']
        self.R = method_param['R']
        self.L = method_param['L']
        self.M = method_param['M']

        self.save_index_name = f'alayalite_index_it_{self.index_type}_qt_{self.quantization_type}_dim_{self.dim}_metric_{self.metric}_M{self.M}.idx'
        print("alaya init done")

    def fit(self, X: np.array) -> None:
        
        if os.path.exists(os.path.join(self.index_save_dir, self.save_index_name)):
            self.index = Index.load(self.index_save_dir, self.save_index_name)
            print('load index from cache')
        else:
            X = X.astype(np.float32)
            self.index = self.client.create_index(name=self.save_index_name, metric=self.metric, quantization_type=self.quantization_type, capacity=X.shape[0])
            self.index.fit(vectors=X, num_threads=self.fit_threads)
            self.client.save_index(self.save_index_name)
            print('save index to cache')

    def set_query_arguments(self, ef):
        self.ef = int(ef)

    def prepare_query(self, q: np.array, n: int):
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.index.search(query=self.q, topk=self.n, ef_search=self.ef)

    def batch_query(self, X: np.array, n: int) -> None:
        self.res = self.index.batch_search(queries=X, topk=n, ef_search=self.ef)

    def get_prepared_query_results(self):
        return self.res

    def get_batch_results(self) -> np.array:
        return self.res

    def __str__(self) -> str:
        return "AlayaLite"

