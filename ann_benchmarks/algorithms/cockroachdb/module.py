"""
This module supports connecting to a CockroachDB instance and performing vector
indexing and search using the built-in C-SPANN vector index.

CockroachDB is started automatically inside the Docker container via the
entrypoint script. The module connects to localhost:26257 as the root user
in insecure mode.

The following environment variables can override connection parameters:

ANN_BENCHMARKS_CRDB_HOST
ANN_BENCHMARKS_CRDB_PORT
ANN_BENCHMARKS_CRDB_USER
ANN_BENCHMARKS_CRDB_DBNAME
"""

import os
import sys
import time

import pgvector.psycopg
import psycopg
from psycopg import sql

from ..base.module import BaseANN


METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<=>",
        "ops_type": "cosine",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "l2",
    },
    "dot": {
        "distance_operator": "<#>",
        "ops_type": "ip",
    },
}


def _get_conn_param(name, default=None):
    val = os.getenv(f"ANN_BENCHMARKS_CRDB_{name.upper()}", default)
    if val is None or len(val.strip()) == 0:
        return default
    return val


class CockroachDB(BaseANN):
    def __init__(self, metric, method_param=None):
        self._metric = metric
        
        if method_param is None:
            method_param = {}
        
        self._build_beam_size = method_param.get("build_beam_size")
        self._min_partition_size = method_param.get("min_partition_size")
        self._max_partition_size = method_param.get("max_partition_size")
        
        self._beam_size = None
        self._cur = None

        if metric not in METRIC_PROPERTIES:
            raise RuntimeError(
                f"unknown metric {metric}, "
                f"expected one of {list(METRIC_PROPERTIES.keys())}"
            )

        op = METRIC_PROPERTIES[metric]["distance_operator"]
        self._query = f"SELECT id FROM items ORDER BY embedding {op} %s LIMIT %s"

    def _connect_with_retry(self, max_wait_sec=30):
        """Connect to CockroachDB, retrying until the server is ready."""
        connect_kwargs = dict(
            host=_get_conn_param("host", "localhost"),
            port=int(_get_conn_param("port", "26257")),
            user=_get_conn_param("user", "root"),
            dbname=_get_conn_param("dbname", "defaultdb"),
            autocommit=True,
        )

        start = time.time()
        last_err = None
        while time.time() - start < max_wait_sec:
            try:
                conn = psycopg.connect(**connect_kwargs)
                print("Connected to CockroachDB")
                return conn
            except psycopg.OperationalError as e:
                last_err = e
                time.sleep(1)

        raise RuntimeError(
            f"Failed to connect to CockroachDB after {max_wait_sec}s: {last_err}"
        )

    def fit(self, X):
        conn = self._connect_with_retry()

        pgvector.psycopg.register_vector(conn)

        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(
            sql.SQL("CREATE TABLE items (id INT PRIMARY KEY, embedding VECTOR({}))")
            .format(sql.Literal(X.shape[1]))
        )

        print("copying data...")
        sys.stdout.flush()
        num_rows = X.shape[0]
        insert_start = time.time()

        with cur.copy(
            "COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int8", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))

        insert_elapsed = time.time() - insert_start
        print(f"inserted {num_rows} rows in {insert_elapsed:.3f} seconds")

        print("creating index...")
        sys.stdout.flush()
        ops_type = METRIC_PROPERTIES[self._metric]["ops_type"]

        with_params = {}
        if self._build_beam_size is not None:
            with_params["build_beam_size"] = self._build_beam_size
        if self._min_partition_size is not None:
            with_params["min_partition_size"] = self._min_partition_size
        if self._max_partition_size is not None:
            with_params["max_partition_size"] = self._max_partition_size

        create_index_sql = sql.SQL(
            "CREATE INDEX ON items USING cspann (embedding {})"
        ).format(sql.Identifier(f"vector_{ops_type}_ops"))

        if with_params:
            params_sql = sql.SQL(", ").join(
                sql.SQL("{} = {}").format(sql.Identifier(k), sql.Literal(v))
                for k, v in with_params.items()
            )
            create_index_sql = sql.SQL("{} WITH ({})").format(
                create_index_sql, params_sql
            )

        cur.execute(create_index_sql)
        print("done!")

        self._cur = cur

    def set_query_arguments(self, beam_size):
        self._beam_size = beam_size
        self._cur.execute(
            sql.SQL("SET vector_search_beam_size = {}").format(sql.Literal(beam_size))
        )

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def done(self):
        if self._cur is not None:
            conn = self._cur.connection
            self._cur.close()
            conn.close()
            self._cur = None

    def __str__(self):
        params = []
        if self._build_beam_size is not None:
            params.append(f"build_beam_size={self._build_beam_size}")
        if self._min_partition_size is not None:
            params.append(f"min_partition_size={self._min_partition_size}")
        if self._max_partition_size is not None:
            params.append(f"max_partition_size={self._max_partition_size}")

        params.append(f"beam_size={self._beam_size}")
        return f"CockroachDB({', '.join(params)})"
