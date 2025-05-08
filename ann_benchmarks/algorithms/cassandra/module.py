import uuid
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement

from ann_benchmarks.algorithms.base.module import BaseANN

class Cassandra(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self.metric = metric
        self.dimension = dimension
        self.method_param = method_param
        self.param_string = "-".join(k + "-" + str(v) for k, v in self.method_param.items()).lower()
        self.index_name = f"os-{self.param_string}"

        self.cluster = Cluster(self.config["host"])
        self.conn = self.cluster.connect()
        self._setup_keyspace()

    def _setup_keyspace(self):
        self.session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
        """)
        self.session.set_keyspace(self.keyspace)

    def _create_table(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id BIGINT PRIMARY KEY,
                embedding VECTOR<FLOAT, {self.dimension}>,
            ) WITH compaction = {{ 'class': 'LeveledCompactionStrategy' }};
        """)
        
    def fit(self, X, batch_size=1000):
        self.vector_dim = X.shape[1]
        self._create_table()

        insert_query = f"INSERT INTO {self.table} (id, embedding) VALUES (?, ?)"
        prepared = self.session.prepare(insert_query)
        batch = BatchStatement()

        for i, vec in enumerate(X):
            batch.add(prepared, (uuid.uuid4(), vec.tolist()))

            if len(batch) >= batch_size:
                self.session.execute(batch)
                batch.clear()
        if batch:
            self.session.execute(batch)

    def query(self, v, n):
        query = f"""
        SELECT id FROM {self.table}
        ORDER BY embedding ANN OF %s
        LIMIT %s
        """
        results = self.session.execute(query, (v.tolist(), n))
        return [row.id for row in results]
   
    def batch_query(self, X, n): 
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res
