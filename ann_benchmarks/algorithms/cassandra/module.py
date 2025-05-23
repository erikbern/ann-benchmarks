import uuid
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, BatchType

from ann_benchmarks.algorithms.base.module import BaseANN

class Cassandra(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self.metric = metric
        self.dimension = dimension
        self.method_param = method_param
        self.keyspace = "ann_benchmarks"
        self.param_string = "-".join(k + "-" + str(v) for k, v in self.method_param.items()).lower()
        self.index_name = f"os-{self.param_string}"
        self.table_name = f"vector_items-{self.param_string}"

        self.cluster = Cluster(['localhost'])
        self.conn = self.cluster.connect()
        self._setup_keyspace()

    def _setup_keyspace(self):
        self.conn.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
        """)
        self.conn.set_keyspace(self.keyspace)

    def _create_table(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id BIGINT PRIMARY KEY,
                embedding VECTOR<FLOAT, {self.dimension}>,
            ) WITH compaction = {{ 'class': 'LeveledCompactionStrategy' }};
        """)

    def _create_index(self, ):
        DISTANCE_MAPPING = {
            "L2": "EUCLIDEAN",
            "COSINE": "COSINE",
            "DOT": "DOT_PRODUCT",
        }

        hnsw_distance_type = DISTANCE_MAPPING.get(self.metric, "EUCLIDEAN") 
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.index_name}
                ON {self.keyspace}.{self.table_name}(embedding) USING 'sai'
                WITH OPTIONS = {{ 'similarity_function': '{hnsw_distance_type}' }};
        """)
        
    def fit(self, X, batch_size=100):
        self.vector_dim = X.shape[1]
        self._create_table()
        self._create_index()

        insert_query = f"INSERT INTO {self.table_name} (id, embedding) VALUES (?, ?)"
        prepared = self.conn.prepare(insert_query)
        batch = BatchStatement(batch_type=BatchType.UNLOGGED)

        for i, vec in enumerate(X):
            batch.add(prepared, (i, vec.tolist()))

            if len(batch) >= batch_size:
                self.conn.execute(batch)
                batch.clear()
        if batch:
            self.conn.execute(batch)

    def set_query_arguments(self, params):
        self._ef_search = params

    def query(self, v, n):
        query = f"""
        SELECT id FROM {self.table_name}
        ORDER BY embedding ANN OF %s
        LIMIT %s
        """
        results = self.conn.execute(query, (v.tolist(), n))
        return [row.id for row in results]
   
    def batch_query(self, X, n): 
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res
