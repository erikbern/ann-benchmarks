import struct
import time

import numpy as np
import psycopg
from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format
from psycopg.types import TypeInfo

from ..base.module import BaseANN


class VectorDumper(Dumper):
    format = Format.BINARY

    def dump(self, obj):
        return struct.pack(f"<H{len(obj)}f", len(obj), *obj)


class VectorLoader(Loader):
    def load(self, buf):
        if isinstance(buf, memoryview):
            buf = bytes(buf)
        dim = struct.unpack_from("<H", buf)[0]
        return np.frombuffer(buf, dtype="<f", count=dim, offset=2)


def register_vector(conn: psycopg.Connection):
    info = TypeInfo.fetch(conn=conn, name="vector")
    register_vector_type(conn, info)


def register_vector_type(conn: psycopg.Connection, info: TypeInfo):
    if info is None:
        raise ValueError("vector type not found")
    info.register(conn)

    class VectorBinaryDumper(VectorDumper):
        oid = info.oid

    adapters = conn.adapters
    adapters.register_dumper(list, VectorBinaryDumper)
    adapters.register_dumper(np.ndarray, VectorBinaryDumper)
    adapters.register_loader(info.oid, VectorLoader)


class PGVectoRS(BaseANN):
    def __init__(self, metric, method_param) -> None:
        self.metric = metric
        self.m = method_param["M"]
        self.ef_construction = method_param["efConstruction"]
        self.ef_search = 100

        if metric == "angular":
            self.query_sql = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
            self.index_sql = f"CREATE INDEX ON items USING vectors (embedding vector_cos_ops) WITH (options = $$[indexing.hnsw]\nm = {self.m}\nef_construction = {self.ef_construction}$$)"
        elif metric == "euclidean":
            self.query_sql = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
            self.index_sql = f"CREATE INDEX ON items USING vectors (embedding vector_l2_ops) WITH (options = $$[indexing.hnsw]\nm = {self.m}\nef_construction = {self.ef_construction}$$)"
        else:
            raise RuntimeError(f"unknown metric {metric}")

        self.connect = psycopg.connect(user="postgres", password="password", autocommit=True)
        self.connect.execute("SET search_path = \"$user\", public, vectors")
        self.connect.execute("CREATE EXTENSION IF NOT EXISTS vectors")
        register_vector(self.connect)

    def fit(self, X):
        dim = X.shape[1]

        cur = self.connect.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(f"CREATE TABLE items (id int, embedding vector({dim}))")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, emb in enumerate(X):
                copy.write_row((i, emb))

        cur.execute(self.index_sql)
        print("waiting for indexing to finish...")
        for _ in range(3600):
            cur.execute("SELECT idx_indexing FROM vectors.pg_vector_index_stat WHERE tablename='items'")
            if not cur.fetchone()[0]:
                break
            time.sleep(10)

    def set_query_arguments(self, ef_search):
        self.ef_search = ef_search
        self.connect.execute(f"SET vectors.hnsw_ef_search = {ef_search}")

    def query(self, vec, num):
        cur = self.connect.execute(self.query_sql, (vec, num), binary=True, prepare=True)
        return [id for (id,) in cur.fetchall()]

    def __str__(self):
        return (
            f"PGVectoRS(metric={self.metric}, m={self.m}, "
            f"ef_construction={self.ef_construction}, ef_search={self.ef_search})"
        )
