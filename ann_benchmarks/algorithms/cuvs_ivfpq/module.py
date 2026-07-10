"""
This module supports connecting to a PostgreSQL instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of PostgreSQL user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If PostgreSQL is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting PostgreSQL
connection parameters:

ANN_BENCHMARKS_PG_USER
ANN_BENCHMARKS_PG_PASSWORD
ANN_BENCHMARKS_PG_DBNAME
ANN_BENCHMARKS_PG_HOST
ANN_BENCHMARKS_PG_PORT

This module starts the PostgreSQL service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_PG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.

Enhanced with cuVS optimization for GPU-accelerated vector operations.
"""

import os
import subprocess
import sys
import threading
import time
import numpy as np

import pgvector.psycopg
import psycopg

# 添加 cuvs 相关导入
from cuvs.neighbors import ivf_pq
import cupy

from typing import Dict, Any, Optional, List, Tuple

from ..base.module import BaseANN
from ...util import get_bool_env_var

# cuVS 距离度量映射
CUVS_METRIC_MAP = {
    "angular": "cosine",
    "euclidean": "euclidean"
}

class CuvsIVFPQ(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._lists = method_param['lists']  # Number of lists for IVFFlat
        self._batch_size = method_param.get('batch_size', 5000)  # 批处理大小
        self._cur = None
        self._conn = None
        
        # cuVS 相关属性
        self._cuvs_index = None
        self._cuvs_vectors = None
        self._cuvs_ids = None
        self._vector_dim = None

    def _build_cuvs_index(self, dataset: np.ndarray) -> None:
        """构建 cuVS 索引"""
        print("Building cuVS index for GPU acceleration...")
        self._vector_dim = dataset.shape[1]
        self._cuvs_vectors = dataset.astype(np.float32)
        self._cuvs_ids = np.arange(len(dataset))
        
        # 构建 cuVS IVF-PQ 索引
        cuvs_metric = CUVS_METRIC_MAP.get(self._metric, "euclidean")
        index_params = ivf_pq.IndexParams(n_lists=self._lists, metric=cuvs_metric)
        self._cuvs_index = ivf_pq.build(index_params, self._cuvs_vectors)
        print(f"cuVS IVF-PQ index built with {self._lists} lists")

    def _cuvs_search(self, query_vector: np.ndarray, k: int) -> List[int]:
        """使用 cuVS 执行单个查询"""
        if self._cuvs_index is None:
            raise RuntimeError("cuVS index not available")
            
        # 将查询向量转换为 GPU 数组
        query_gpu = cupy.asarray(np.array([query_vector], dtype=np.float32))
        
        # 执行搜索
        search_params = ivf_pq.SearchParams(n_probes=self._probes)
        distances, indices = ivf_pq.search(search_params, self._cuvs_index, query_gpu, k)
        
        # 转换回 CPU
        indices_cpu = cupy.asnumpy(indices)
        
        # 返回对应的 ID
        result_ids = [self._cuvs_ids[idx] for idx in indices_cpu[0]]
        return result_ids

    def _cuvs_batch_search(self, query_vectors: np.ndarray, k: int) -> List[List[int]]:
        """使用 cuVS 执行批量查询 - 真正的 GPU 批量处理"""
        if self._cuvs_index is None:
            raise RuntimeError("cuVS index not available")
            
        print(f"Performing cuVS batch search for {len(query_vectors)} queries...")
        
        # 将查询向量转换为 GPU 数组
        queries_gpu = cupy.asarray(query_vectors.astype(np.float32))
        
        # 执行批量搜索
        search_params = ivf_pq.SearchParams(n_probes=self._probes)
        distances, indices = ivf_pq.search(search_params, self._cuvs_index, queries_gpu, k)
        
        # 转换回 CPU
        indices_cpu = cupy.asnumpy(indices)
        
        # 返回对应的 ID 列表
        results = []
        for i in range(len(query_vectors)):
            result_ids = [self._cuvs_ids[idx] for idx in indices_cpu[i]]
            results.append(result_ids)
        
        print(f"Batch search completed for {len(query_vectors)} queries")
        return results



    def fit(self, dataset):
        if dataset.shape[0] > 1000000:
            self._lists = int(np.sqrt(dataset.shape[0]))
        else:
            self._lists = int(dataset.shape[0]/1000)
        # 构建 cuVS 索引
        self._build_cuvs_index(dataset)
        print("cuVS GPU acceleration initialized")

    def set_query_arguments(self, probes):
        self._probes = probes
        print(f"Set cuVS search probes to {probes}")

    def query(self, v, n):
        """单次查询，使用 cuVS GPU 加速"""
        return self._cuvs_search(v, n)

    def batch_query(self, X, n):
        """执行批量查询，使用 cuVS GPU 加速"""
        self._batch_results = self._cuvs_batch_search(X, n)

    def get_batch_results(self):
        """获取批量查询的结果"""
        return self._batch_results

    def get_memory_usage(self):# cuvs里面有没有类似的监控方法
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __del__(self):
        """清理资源"""
        # cuVS 会自动清理 GPU 资源
        pass

    def __str__(self):
        return f"CuvsIVFPQ(lists={self._lists}, probes={self._probes})"