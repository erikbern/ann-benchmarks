import os
import numpy as np
import time

from typing import Iterable, List, Any
from enum import Enum
from pythonping import ping

from aerospike_vector import vectordb_admin, vectordb_client, types, types_pb2

from ..base.module import BaseANN

'''


// Maximum number bi-directional links per HNSW vertex. Greater values of
  // 'm' in general provide better recall for data with high dimensionality, while
  // lower values work well for data with lower dimensionality.
  // The storage space required for the index increases proportionally with 'm'.
  // The default value is 16.
  optional uint32 m = 1;

  // The number of candidate nearest neighbors shortlisted during index creation.
  // Larger values provide better recall at the cost of longer index update times.
  // The default is 100.
  optional uint32 efConstruction = 2;

  // The default number of candidate nearest neighbors shortlisted during search.
  // Larger values provide better recall at the cost of longer search times.
  // The default is 100.
  optional uint32 ef = 3;
'''
_AerospikeIdxNames : list = []

class VectorDistanceMetric(Enum):
        SQUARED_EUCLIDEAN = types_pb2.VectorDistanceMetric.SQUARED_EUCLIDEAN
        COSINE = types_pb2.VectorDistanceMetric.COSINE
        DOT_PRODUCT = types_pb2.VectorDistanceMetric.DOT_PRODUCT
        MANHATTAN = types_pb2.VectorDistanceMetric.MANHATTAN
        HAMMING = types_pb2.VectorDistanceMetric.HAMMING

class Aerospike(BaseANN):
    
    def __init__(self,
                    metric: str, 
                    dimension: int,
                    idx_type,
                    hnswParams: dict,
                    uniqueIdxName: bool = True,
                    dropIdx: bool = True,
                    ping: bool = False):
        self._metric = metric
        self._dims = dimension
        self._idx_type = idx_type.upper()        
        self._idx_value = VectorDistanceMetric[self._idx_type].value
        
        if hnswParams is None or len(hnswParams) == 0:
            self._idx_hnswparams = None
        else:
            self._idx_hnswparams = Aerospike.SetHnswParamsAttrs(
                                        types_pb2.HnswParams(),
                                        hnswParams
                                    )
        self._idx_drop = dropIdx
        self._username = os.environ.get("APP_USERNAME") or ""
        self._password = os.environ.get("APP_PASSWORD") or ""
        self._host = os.environ.get("PROXIMUS_HOST") or "localhost"
        self._port = int(os.environ.get("PROXIMUS_PORT") or 5000)
        self._listern = os.environ.get("PROXIMUS_ADVERTISED_LISTENER") or None          
        self._namespace = os.environ.get("PROXIMUS_NAMESPACE") or "test"
        self._setName = os.environ.get("PROXIMUS_SET") or "ANN-data"
        self._verifyTLS = os.environ.get("VERIFY_TLS") or True
        self._idx_parallism = int(os.environ.get("APP_INDEXER_PARALLELISM") or 1)
        if not uniqueIdxName or self._idx_hnswparams is None:
            self._idx_name = f'{self._setName}_{self._idx_type}_Idx'
        else:
            self._idx_name = f'{self._setName}_{self._idx_type}_{self._dims}_{self._idx_hnswparams.m}_{self._idx_hnswparams.efConstruction}_{self._idx_hnswparams.ef}_Idx'
        self._idx_binName = "ANN_embedding"
        self._idx_binKeyName = "ANN_key"
        self._query_hnswsearchparams = None
        
        if ping:
            print(f'Aerospike: Trying Connection to {self._host} {self._verifyTLS} {self._listern}')
            print(ping(self._host, verbose=True))
        print('Aerospike: Try Create Admin client')
        
        self._adminClient = vectordb_admin.VectorDbAdminClient(
                                types.HostPort(self._host,
                                                self._port,
                                                self._verifyTLS), 
                                self._listern)
       
        self._client = vectordb_client.VectorDbClient(
                            types.HostPort(self._host,
                                            self._port,
                                            self._verifyTLS), 
                            self._listern)
    
    @staticmethod
    def SetHnswParamsAttrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batchingParams':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            types_pb2.HnswBatchingParams(),
                            __dict[key].asdict()
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj
        
    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        print(f'done: {self}')
        
        if self._client is not None:
            self._client.close()
        if self._adminClient is not None:
            self._adminClient.close()
        
    def fit(self, X: np.array) -> None:
        global _AerospikeIdxNames
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
                
        print(f'Aerospike fit: {self} Shape: {X.shape}')
        
        populateIdx = True
        
        #If exists, no sense to try creation...
        if(any(index.id.namespace == self._namespace
                                and index.id.name == self._idx_name 
                        for index in self._adminClient.indexList())):
            print(f'Aerospike: Index {self._namespace}.{self._idx_name} Already Exists')
            
            #since this can be an external DB (not in a container), we need to clean up from prior runs
            #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
            #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
            if self._idx_name in _AerospikeIdxNames:
                print(f'Aerospike: Index {self._namespace}.{self._idx_name} being reused (not re-populated)')
                populateIdx = False
            elif self._idx_drop:
                print(f'Aerospike: Dropping Index...')
                s = time.time()
                self._adminClient.indexDrop(namespace=self._namespace,
                                            name=self._idx_name)
                t = time.time()
                print(f"Aerospike: Drop Index Time (sec) = {t - s}")                
            else:
                populateIdx = False
        else:
            print(f'Aerospike: Creating Index {self._namespace}.{self._idx_name}')
            s = time.time()
            self._adminClient.indexCreate(
                                namespace=self._namespace,
                                name=self._idx_name,
                                setFilter=self._setName,
                                vector_bin_name=self._idx_binName,
                                dimensions=self._dims,
                                indexParams= self._idx_hnswparams,
                                vector_distance_metric=self._idx_value,
                            )
            t = time.time()
            print(f"Aerospike: Index Creation Time (sec) = {t - s}")
            _AerospikeIdxNames.append(self._idx_name)
            
        if populateIdx:
            print(f'Aerospike: Populating Index {self._namespace}.{self._idx_name}')
            s = time.time()
            for i, embedding in enumerate(X):
                #print(f'Item {i},{embedding.shape}  Vector:{embedding}')
                self._client.put(namespace=self._namespace,
                                    set=self._setName,                            
                                    key=i,
                                    bins={
                                    self._idx_binName:embedding.tolist(),
                                    self._idx_binKeyName:i
                                    },
                )
            t = time.time()
            print(f"Aerospike: Index Put Time (sec) = {t - s}")
            print("Aerospike: waiting for indexing to complete")
            self._client.waitForIndexCompletion(self._namespace, self._idx_name)
            t = time.time()
            print(f"Aerospike: Index Total Populating Time (sec) = {t - s}")
    
    def set_query_arguments(self, hnswParams: dict = None):
        if hnswParams is not None and len(hnswParams) > 0:
            self._query_hnswsearchparams = Aerospike.SetHnswParamsAttrs(
                                                    types_pb2.HnswSearchParams(),
                                                    hnswParams
                                                )
           
    def query(self, q, n):
        result = self._client.vectorSearch(self._namespace,
                                                self._idx_name,
                                                q.tolist(),
                                                n,
                                                self._query_hnswsearchparams,
                                                self._idx_binKeyName)
        result_ids = [neighbor.bins[self._idx_binKeyName] for neighbor in result]
        return result_ids
    
    #def get_batch_results(self):
    #    return self.batch_results

    #def get_batch_latencies(self):
    #    return self.batch_latencies

    def __str__(self):
        hnswparams = str(self._idx_hnswparams).strip().replace("\n", ", ")
        return f"Aerospike([{self._metric}, {self._host}, {self._port}, {self._namespace}, {self._setName}, {self._idx_name}, {self._idx_type}, {self._idx_value}, {self._dims}, {{{hnswparams}}}, {{{self._query_hnswsearchparams}}}])"
