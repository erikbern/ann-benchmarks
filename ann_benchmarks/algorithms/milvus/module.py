from time import sleep
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import os

from ..base.module import BaseANN


def metric_mapping(_metric: str):
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.start_milvus()
        self.connects = connections
        max_trys = 10
        for try_num in range(max_trys):
            try:
                self.connects.connect("default", host='localhost', port='19530')
                break
            except Exception as e:
                if try_num == max_trys - 1:
                    raise Exception(f"[Milvus] connect to milvus failed: {e}!!!")
                print(f"[Milvus] try to connect to milvus again...")
                sleep(1)
        print(f"[Milvus] Milvus version: {utility.get_server_version()}")
        self.collection_name = "test_milvus"
        if utility.has_collection(self.collection_name):
            print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
            utility.drop_collection(self.collection_name)

    def start_milvus(self):
        try:
            os.system("docker compose down")
            os.system("docker compose up -d")
            print("[Milvus] docker compose up successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose up failed: {e}!!!")

    def stop_milvus(self):
        try:
            os.system("docker compose down")
            print("[Milvus] docker compose down successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose down failed: {e}!!!")

    def create_collection(self):
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        filed_vec = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )
        schema = CollectionSchema(
            fields=[filed_id, filed_vec],
            description="Test milvus search",
        )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Milvus] Create collection {self.collection.describe()} successfully!!!")

    def insert(self, X):
        # insert data
        print(f"[Milvus] Insert {len(X)} data into collection {self.collection_name}...")
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_data = X[i: min(i + batch_size, len(X))]
            entities = [
                [i for i in range(i, min(i + batch_size, len(X)))],
                batch_data.tolist()
            ]
            self.collection.insert(entities)
        self.collection.flush()
        print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

    def get_index_param(self):
        raise NotImplementedError()

    def create_index(self):
        # create index
        print(f"[Milvus] Create index for collection {self.collection_name}...")
        self.collection.create_index(
            field_name = "vector",
            index_params = self.get_index_param(),
            index_name = "vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        index = self.collection.index(index_name = "vector_index")
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        print(f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")

    def load_collection(self):
        # load collection
        print(f"[Milvus] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")

    def fit(self, X):
        self.create_collection()
        self.insert(X)
        self.create_index()
        self.load_collection()

    def query(self, v, n):
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

    def done(self):
        self.collection.release()
        utility.drop_collection(self.collection_name)
        self.stop_milvus()


class MilvusFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(self, v, n):
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class MilvusIVFFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFSQ8(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFPQ(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }
    
    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFPQ metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusHNSW(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, ef):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"MilvusHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class MilvusSCANN(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"
