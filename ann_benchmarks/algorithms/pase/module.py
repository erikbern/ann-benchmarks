"""
This module supports connecting to a PostgreSQL instance and performing vector
indexing and search using the PASE extension. The default behavior uses
the "ann" value of PostgreSQL user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If PostgreSQL is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting PostgreSQL
connection parameters:

ANN_BENCHMARKS_PG_USER
ANN_BENCHMARKS_PG_PASSWORD
ANN_BENCHMARKS_PG_DBNAME
ANN_BENCHMARKS_PG_HOST
ANN_BENCHMARKS_PG_PORT

This module starts the PostgreSQL service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_PG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the PASE extension inside the
target database, if it has not been already created.
"""

import subprocess
import sys
import os
import time
import numpy as np

import psycopg

from typing import Dict, Any, Optional, List

from ..base.module import BaseANN
from ...util import get_bool_env_var
from psycopg.types import array


def get_pg_param_env_var_name(pg_param_name: str) -> str:
    """
    Generate the environment variable name for PostgreSQL connection parameters.

    Args:
        pg_param_name (str): The name of the PostgreSQL parameter.

    Returns:
        str: The corresponding environment variable name.
    """
    return f'ANN_BENCHMARKS_PG_{pg_param_name.upper()}'


def get_pg_conn_param(
        pg_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    """
    Retrieve PostgreSQL connection parameter from environment variable.

    Args:
        pg_param_name (str): The name of the PostgreSQL parameter.
        default_value (Optional[str], optional): Default value if not set. Defaults to None.

    Returns:
        Optional[str]: The parameter value from environment or default.
    """
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value


class PASE(BaseANN):
    """
    PASE (Parallel Approximate Similarity Search Extension) implementation
    for Approximate Nearest Neighbor benchmarking.
    """

    def __init__(self, metric: str, method_param: Dict[str, Any]):
        """
        Initialize PASE algorithm parameters.

        Args:
            metric (str): Distance metric to use ('angular' or 'euclidean').
            method_param (Dict[str, Any]): Method-specific parameters.
        """
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._ef_search = method_param.get('efSearch', 40)  # Default to 40 if not specified
        self._cur = None

    def ensure_pase_extension_created(self, conn: psycopg.Connection) -> None:
        """
        Ensure that the PASE extension is created in the database.

        Args:
            conn (psycopg.Connection): PostgreSQL database connection.
        """
        with conn.cursor() as cur:
            # Check if the extension exists
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pase')")
            pase_exists = cur.fetchone()[0]

            if pase_exists:
                print("PASE extension already exists")
            else:
                print("PASE extension does not exist, creating")
                cur.execute("CREATE EXTENSION pase")
                conn.commit()
                print("Successfully created PASE extension")

    def fit(self, X):
        """
        Fit the PASE index with the given dataset.

        Args:
            X (Any): The input dataset of vectors.
        """
        # Prepare connection parameters with defaults
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')

        # Always use /tmp for the socket directory
        psycopg_connect_kwargs['host'] = '/tmp'

        pg_port_str: Optional[str] = get_pg_conn_param('port')
        if pg_port_str is not None:
            psycopg_connect_kwargs['port'] = int(pg_port_str)

        # Decide whether to start the PostgreSQL service
        should_start_service = get_bool_env_var(
            get_pg_param_env_var_name('start_service'),
            default_value=True)
        if should_start_service:
            # Start PostgreSQL using pg_ctl as postgres user
            try:
                # First check if PostgreSQL is already running
                status = subprocess.run(
                    "su postgres -c '/usr/local/pgsql/bin/pg_ctl status -D /usr/local/pgsql/data'",
                    shell=True,
                    capture_output=True
                )

                if status.returncode != 0:  # Not running, so start it
                    print("Starting PostgreSQL server...")
                    subprocess.run(
                        "su postgres -c '/usr/local/pgsql/bin/pg_ctl start -D /usr/local/pgsql/data -l /usr/local/pgsql/logs/logfile -w'",
                        shell=True,
                        check=True,
                        stdout=sys.stdout,
                        stderr=sys.stderr
                    )
                    # Wait a bit for the server to be ready
                    time.sleep(2)
                else:
                    print("PostgreSQL server is already running")
            except subprocess.CalledProcessError as e:
                print(f"Error starting PostgreSQL: {e}")
                raise
        else:
            print(
                "Assuming that PostgreSQL service is managed externally. "
                "Not attempting to start the service.")

        # Establish connection and create PASE extension
        conn = psycopg.connect(**psycopg_connect_kwargs)
        self.ensure_pase_extension_created(conn)

        # Register vector type and create cursor
        array.register_default_adapters(conn.adapters)
        cur = conn.cursor()

        print("X shape:", X.shape[1], X.shape)
        # Prepare table and index
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(f"CREATE TABLE items (id int, embedding float4[%d])" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")

        print("Copying data...")
        # Use a different approach to insert data in batches
        # This avoids the numpy array boolean evaluation issue
        batch_size = 1000  # Process in smaller batches
        print("Copying data using COPY command...")
        total_rows = len(X)

        # Use COPY for faster data loading
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT binary)") as copy:
            copy.set_types(["int4", "float4[]"])
            for i, embedding in enumerate(X):
                # Convert NumPy array to list if needed
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                copy.write_row((i, embedding_list))

                # Print progress every 10000 rows
                if i > 0 and i % 10000 == 0:
                    print(f"Copied {i}/{total_rows} vectors...")

        print("Creating index...")
        
        # Note: PASE uses slightly different index creation syntax
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX pase_hnsw_idx ON items USING pase_hnsw(embedding) WITH (dim = %d, base_nb_num = %d, ef_build = %d)"
                % (X.shape[1], self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            cur.execute(
                "CREATE INDEX pase_hnsw_idx ON items USING pase_hnsw(embedding) WITH (dim = %d, base_nb_num = %d, ef_build = %d)"
                % (X.shape[1], self._m, self._ef_construction)
            )

        print("Indexing complete!")
        self._cur = cur

    def set_query_arguments(self, ef_search: int) -> None:
        """
        Set query-time search parameters.

        Args:
            ef_search (int): Effective search parameter.
        """
        self._ef_search = ef_search

    def query(self, v: Any, n: int) -> List[int]:
        """
        Perform nearest neighbor search.

        Args:
            v (Any): Query vector.
            n (int): Number of neighbors to retrieve.

        Returns:
            List[int]: List of neighbor IDs.
        """
        # Convert NumPy array to list if needed
        if isinstance(v, np.ndarray):
            v = v.tolist()
        # Cast the vector to float4[] explicitly to match the column type
        self._cur.execute(f"SET hnsw.ef_search = {self._ef_search}")
        # Force PostgreSQL to use the index by setting costs
        self._cur.execute("SET enable_seqscan = off")

        if self._metric == "angular":
            query = """SELECT id FROM items ORDER BY embedding <?> pase(ARRAY[%s]::float4[],0,1) LIMIT %s"""
        elif self._metric == "euclidean":
            query = """SELECT id FROM items ORDER BY embedding <?> pase(ARRAY[%s]::float4[],0,0) LIMIT %s"""
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

        self._cur.execute(query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self) -> float:
        """
        Get memory usage of the index.

        Returns:
            float: Memory usage in KB.
        """
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('pase_hnsw_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self) -> str:
        """
        String representation of the PASE configuration.

        Returns:
            str: Formatted configuration string.
        """
        return f"PASE(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"