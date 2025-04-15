import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN


class PGDiskANN(BaseANN):
    def __init__(self, metric, nbits, method_param):
        print(f"running constructor")
        self._metric = metric
        self._nbits = nbits
        self._cur = None
        self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        self._num_neighbors = method_param["num_neighbors"]
        self._search_list_size = method_param["search_list_size"]
        self._max_alpha = method_param["max_alpha"]
        print(f"running only {self._metric} and {self._query}")

    def fit(self, X):
        print("running fit")
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        cur.execute(
            "CREATE INDEX ON items USING diskann(embedding) WITH (num_neighbors = %d, search_list_size = %d, max_alpha = %d, num_bits_per_dimension = %d)"
            % (self._num_neighbors, self._search_list_size, self._max_alpha, self._nbits)
        )
        print("done!")
        self._cur = cur

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def set_query_arguments(self, list_size):
        self._list_size = list_size
        self._cur.execute("SET diskann.query_search_list_size = %d" % list_size)

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGDiskANN(metric={self._metric}), num_neighbors={self._num_neighbors}, search_list_size={self._search_list_size}, max_alpha={self._max_alpha}), nbits={self._nbits})"
