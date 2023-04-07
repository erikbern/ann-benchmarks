import subprocess
import sys

import psycopg2
import pgvector.psycopg2

from ann_benchmarks.algorithms.base import BaseANN

class PGVector(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._lists = lists

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg2.connect(host="localhost", user="ann", password="ann", dbname="ann")
        cur = conn.cursor()
        pgvector.psycopg2.register_vector(conn)
        cur.execute("create table items (item int, embedding vector(%s))", (X.shape[1],))
        for i, embedding in enumerate(X):
            cur.execute('INSERT INTO items (item, embedding) VALUES (%s, %s)', (i, embedding,))
        cur.execute('CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %s)', (self._lists,))
