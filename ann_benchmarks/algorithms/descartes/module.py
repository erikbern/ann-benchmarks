import py01ai
import numpy as np
from ..base.module import BaseANN
import logging

class fng(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "angular", "euclidean": "square_l2"}[metric]
        self.method_param = method_param
        self.name = "01ai (%s)" % (self.method_param)

    def fit(self, X):
        self.index = py01ai.FNGIndex(metric=self.metric, dimension=len(X[0]))
        self.index.init(count=len(X), 
                        m=self.method_param["M"], 
                        s=self.method_param["S"],
                        l=self.method_param["L"])
        try: 
            self.index.add_vector(input=np.asarray(X))
        except Exception as e:
            print(str(e))

    def set_query_arguments(self, ex, ef):
        self.search_ef = ef
        self.search_ex = ex
        self.index.set_query_param(ef=ef, ex=ex)

    def query(self, v, n):
        return self.index.search(query=v, topk=n)

    def freeIndex(self):
        del self.index

    def __str__(self):
        m  = self.method_param['M']
        s  = self.method_param['S']
        l  = self.method_param['L']
        ef = self.search_ef
        ex = self.search_ex
        return "01ai(M=%s S=%s L=%s EF=%s EX=%s)" % (m, s, l, ef, ex)
