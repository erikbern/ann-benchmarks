import subprocess
import sys

import psycopg
import pgvector.psycopg

from .base import BaseANN


class PGVector(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._lists = lists
        self._cur = None

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(host="localhost", user="ann", password="ann", dbname="ann")
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %d)" % self._lists
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = %d)" % self._lists)
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def query(self, v, n):
        if self._metric == "angular":
            q = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif self._metric == "euclidean":
            q = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

        self._cur.execute(q, (v, n))
        return [id for id, in self._cur.fetchall()]

    def __str__(self):
        return f"PGVector(lists={self._lists})"
