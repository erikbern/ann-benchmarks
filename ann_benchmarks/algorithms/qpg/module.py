import os
import struct
import subprocess
import time
import importlib
import multiprocessing

import qpgpy_search
import qpgpy_build
from sklearn import preprocessing
import numpy as np
from ..base.module import BaseANN


def metric_mapping(metric):
    mapping_dict = {"ip": "ip", "euclidean": "L2", "angular": "angular"}
    metric_type = mapping_dict.get(metric)
    if metric_type == "ip":
        return qpgpy_search.ComputerType.IP
    elif metric_type == "angular":
        return qpgpy_search.ComputerType.ANGULAR
    elif metric_type == "L2":
        return qpgpy_search.ComputerType.L2

def metric_mapping_build(metric):
    mapping_dict = {"ip": "ip", "euclidean": "L2", "angular": "angular"}
    metric_type = mapping_dict.get(metric)
    if metric_type == "ip":
        return qpgpy_build.distanceMetric.IP
    elif metric_type == "angular":
        return qpgpy_build.distanceMetric.ANGULAR
    elif metric_type == "L2":
        return qpgpy_build.distanceMetric.L2

def data_type_mapping(data_type):
    if data_type == "Float":
        return qpgpy_search.DataType.FLOAT
    elif data_type == "Int8":
        return qpgpy_search.DataType.UNIT8
    elif data_type == "Int16":
        return qpgpy_search.DataType.UINT16

def rerank_mapping(rerank):
    if rerank == "sq16":
        return qpgpy_search.QuantizerType.SQ16
    if rerank == "sq8":
        return qpgpy_search.QuantizerType.SQ8
    if rerank == "tc8":
        return qpgpy_search.QuantizerType.TC8
    if rerank =="None":
        return qpgpy_search.QuantizerType.NONE

def quantizer_mapping(quantizer):
    if quantizer == "sq16":
        return qpgpy_search.QuantizerType.SQ16
    if quantizer == "sq8":
        return qpgpy_search.QuantizerType.SQ8
    if quantizer == "sq4":
        return qpgpy_search.QuantizerType.SQ4
    if quantizer =="None":
        return qpgpy_search.QuantizerType.NONE

def l2_normalize(data):
    row_norms = np.linalg.norm(data, axis=1, keepdims=True)  # 计算每行的 L2 范数
    return data / row_norms

class QPG(BaseANN):
    def __init__(self, metric, data_type, param):
        self.metric_build = metric_mapping_build(metric)
        self.metric = metric_mapping(metric)
        self.distance_type = metric
        self.data_path = param["data_path"]
        self.data_type = data_type_mapping(data_type)
        self.K = int(param["K"])
        self.smp = int(param["smp"])
        self.iter = int(param["iter"])
        self.alpha = float(param["alpha"])
        self.M0 = int(param["M0"])
        self.M = int(param["M"])
        self.occ = int(param["occ"])
        self.recallN = int(param["recallN"])
        self.rvsN = int(param["rvsN"])
        self.maxThreadNum = multiprocessing.cpu_count() 
        self.useShortCut = 0
        self.out_path = param["out_path"]
        self.reranker = rerank_mapping(param["reranker"])
        self.quantizer = quantizer_mapping(param["quantizer"])
        self.rdim = int(param["rdim"])
        self.qdim = int(param["qdim"])
        self.graph_path = param["out_path"] 


    def fit(self, X):
        data_path=self.data_path
        if isinstance(X, np.ndarray):
            shape = X.shape
            if self.distance_type == "angular":
                X = l2_normalize(X)
            with open(data_path, 'wb') as fvecs_file:
                for vec in X:
                    dim = np.array([len(vec)], dtype=np.uint32)  
                    fvecs_file.write(dim.tobytes())         
                    vec.tofile(fvecs_file)
            
        elif isinstance(X, list):
            train_combined = np.concatenate(X, axis=0)
            train_combined.tofile(data_path)
            print("data is a list")
            print(f"Train data saved as {data_path}")

        bp=qpgpy_build.BuildParam()
        bp.K = self.K
        bp.smpN = self.smp
        bp.iterN = self.iter
        bp.recallN = self.recallN
        bp.rvsN = self.rvsN
        bp.maxThreadNum = self.maxThreadNum
        bp.useShortCut = self.useShortCut
        bp.dist_metric = self.metric_build
        bp.gType = qpgpy_build.GraphType.XNDesc

        tp=qpgpy_build.TSDGPara()
        tp.maxK_ = self.M0
        tp.alpha_ = self.alpha
        tp.occThres_ = self.occ
        tp.dist_metric = self.metric_build

        M = self.M
        in_data_pth = data_path
        out_pth = self.out_path
        reorder_tag = 0
        qpgpy_build.buildHTSDG(bp,tp,M,in_data_pth,out_pth,reorder_tag)
  

    def set_query_arguments(self, parameters):
        efs,expand = parameters
        p=qpgpy_search.Paths()
        p.data_path = self.data_path
        p.out_path = self.out_path
        p.graph_path = self.graph_path
        
        sc=qpgpy_search.searcher_config()
        sc.orig_type = self.data_type
        sc.data_type = self.data_type
        sc.quan_type = self.data_type
        sc.rerank = self.reranker
        sc.qtizer = self.quantizer
        sc.metric = self.metric
        
        pm=qpgpy_search.search_params()
        pm.quantizer_dim = self.qdim
        pm.reranker_dim = self.rdim
        pm.k = 10
        pm.efsearch_expand = expand

        self.name = "QPG(%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%f)" % (
            self.K,
            self.smp,
            self.iter,
            self.alpha,
            self.occ,
            self.M0,
            self.M,
            self.qdim,
            self.rdim,
            efs,
            expand
            
        )
        
        self.nns=qpgpy_search.initialize_search(p, sc, pm, efs, expand)
        
    
    def batch_query(self,v,n):
        if self.distance_type == "angular":
                v = l2_normalize(v)
        self.res=qpgpy_search.batch_search(self.nns,v,n,v.shape[1],v.shape[0])
        


    def freeIndex(self):
        print("QPG: free")