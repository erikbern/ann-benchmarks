from __future__ import absolute_import
import time
import milvus
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class MilvusIVFFLAT(BaseANN):
    def __init__(self, metric, index_type, nlist):
        self._index_param = {'nlist': nlist}
        self._search_param = {'nprobe': None}
        self._metric = metric
        self._milvus = milvus.Milvus(host='localhost', port='19530', try_connect=False, pre_ping=False)
        self._table_name = 'test01'
        self._index_type = index_type

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        self._milvus.create_collection({'collection_name': self._table_name, 'dimension': X.shape[1], 'index_file_size': 2048})
        vector_ids = [id for id in range(len(X))]
        records = X.tolist()
        records_len = len(records)
        step = 100000
        for i in range(0, records_len, step):
            end = min(i + step, records_len)
            status, ids = self._milvus.insert(collection_name=self._table_name, records=records[i:end], ids=vector_ids[i:end])
            if not status.OK():
                raise Exception("Insert failed. {}".format(status))
        self._milvus.flush([self._table_name])

        while True:
            status, stats = self._milvus.get_collection_stats(self._table_name)
            if len(stats["partitions"][0]["segments"]) > 1:
                time.sleep(2)
            else:
                break

        index_type = getattr(milvus.IndexType, self._index_type)  # a bit hacky but works
        status = self._milvus.create_index(self._table_name, index_type, params=self._index_param)
        if not status.OK():
            raise Exception("Create index failed. {}".format(status))
#         self._milvus_id_to_index = {}
#         self._milvus_id_to_index[-1] = -1 #  -1 means no results found
#         for i, id in enumerate(ids):
#             self._milvus_id_to_index[id] = i

    def set_query_arguments(self, nprobe):
        if nprobe > self._index_param['nlist']:
            print('warning! nprobe > nlist')
            nprobe = self._index_param['nlist']
        self._search_param['nprobe'] = nprobe

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        v = v.tolist()
        future = self._milvus.search(collection_name=self._table_name, query_records=[v], top_k=n, params=self._search_param, _async=True)
        return future

    def handle_query_list_result(self, query_list):
        handled_result = []
        t0 = time.time()
        for index, query in enumerate(query_list):
            total, v, future = query
            status, results = future.result()
            if not status.OK():
                raise Exception("[Search] search failed: {}".format(status.message))

            if not results:
                raise Exception("Query result is empty")
            # r = [self._milvus_id_to_index[z.id] for z in results[0]]
            results_ids = []
            for result in results[0]:
                results_ids.append(result.id)
            handled_result.append((total, v, results_ids))
            # return results_ids
        return time.time() - t0, handled_result

    def __str__(self):
        return 'Milvus(index={}, index_param={}, search_param={})'.format(self._index_type, self._index_param, self._search_param)
