import getpass
import glob
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import mariadb

from ..base.module import BaseANN

def vector_to_hex(v):
    binary_data = bytearray(v.size * 4)
    for index, f in enumerate(v):
        struct.pack_into('f', binary_data, index * 4, f)
    return binary_data

class MariaDB(BaseANN):

    def __init__(self, metric, method_param):
        self._test_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._metric = metric
        self._m = method_param['M']
        self._engine = method_param['engine']
        self._cur = None
        self._perf_proc = None
        self._perf_records = []
        self._perf_stats = []


        if metric == "angular":
            raise RuntimeError(f"Angular metric is not supported.")
        elif metric == "euclidean":
            # euclidean is the current default and only distance metric supported by MariaDB
            pass
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
        self.prepare_options()
        self.initialize_db()
        self.start_db()

        # Connect to MariaDB using Unix socket
        conn = mariadb.connect(unix_socket=self._socket_file)
        self._cur = conn.cursor()

    def prepare_options(self):
        self._perf_stat = os.environ.get('PERF', 'no') == 'yes' and MariaDB.can_run_perf()
        self._perf_record = os.environ.get('FLAMEGRAPH', 'no') == 'yes' and MariaDB.can_run_flamegraph()

        if self._perf_stat and self._perf_record:
            self._perf_stat = False
            print("\nWarning: Better not to enable both PERF and FLAMEGRAPH. Generating a flame graph only.\n")

        # MariaDB build dir or installed dir
        mariadb_root_dir = os.environ.get('MARIADB_ROOT_DIR')
        if mariadb_root_dir is None:
            print("MariaDB path MARIADB_ROOT_DIR is not provided. It can be your local build dir or installation dir. "
                  "For local build dir, you also need to specify the MARIADB_SOURCE_DIR for database initialization.")
            raise RuntimeError(f"Could not initialize database.")

        # mariadb-install-db needs `--srcdir` option for local builds
        mariadb_source_dir = os.environ.get('MARIADB_SOURCE_DIR')

        # Initialize database when running locally
        self._do_init_mariadb = os.environ.get('DO_INIT_MARIADB', 'yes') == 'yes'

        # DB workfolder: data + error log
        mariadb_db_workspace = os.environ.get('MARIADB_DB_WORKSPACE')
        if mariadb_db_workspace is None:
            raise RuntimeError("Please specify path MARIADB_DB_WORKSPACE to define the database directory.")
        data_dir = mariadb_db_workspace + '/data'
        log_file = mariadb_db_workspace + '/mariadb.err'
        # Create data directory if not exist
        os.makedirs(f"{data_dir}", exist_ok=True)

        # Generate a socket file name under /tmp to make sure the file path is always under 107 character, to avoid "The socket file path is too long" error
        self._socket_file = tempfile.mktemp(prefix='mysql_', suffix='.sock', dir='/tmp')

        print("\nSetup paths:")
        print(f"MARIADB_ROOT_DIR: {mariadb_root_dir}")
        print(f"DATA_DIR: {data_dir}")
        print(f"LOG_FILE: {log_file}")
        print(f"SOCKET_FILE: {self._socket_file}\n")

        # Command for MariaDB initialization
        self._mariadb_init_cmd = [
            glob.glob(f"{mariadb_root_dir}/*/mariadb-install-db")[0],
            "--no-defaults",
            "--verbose",
            "--skip-name-resolve",
            "--skip-test-db",
            f"--datadir={data_dir}"
        ]
        if mariadb_source_dir is not None:
            self._mariadb_init_cmd.append(f"--srcdir={mariadb_source_dir}")

        # Command for starting MariaDB server
        self._mariadb_start_cmd = [
            glob.glob(f"{mariadb_root_dir}/*/mariadbd")[0],
            "--no-defaults",
            f"--datadir={data_dir}",
            f"--log_error={log_file}",
            f"--socket={self._socket_file}",
            "--skip_networking",
            "--skip_grant_tables",
            f"--mhnsw_max_edges_per_node={self._m}"
        ]
        user_option = MariaDB.get_user_option()
        if user_option is not None:
            self._mariadb_start_cmd += user_option
        self._mariadbd_proc = None

    def initialize_db(self):
        try:
            # In ann-benchmarks build, the server was initialized in Docker image, but when running locally we want to start it with a new initialization
            if self._do_init_mariadb:
                print("\nInitialize MariaDB database...")
                print(self._mariadb_init_cmd)
                #subprocess.run(self._mariadb_init_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
                init_proc = subprocess.Popen(self._mariadb_init_cmd, stdout=sys.stdout, stderr=sys.stderr)
                init_proc.wait()
        except Exception as e:
            print("ERROR: Failed to initialize MariaDB database:", e)
            raise

    @staticmethod
    def get_user_option():
        # to support running with root user
        try:
            return ["--user=root"] if getpass.getuser() == "root" else None
        except Exception as e:
            print("Could not get current user, could be docker user mapping. Ignore.")
            return None

    @staticmethod
    def can_run_perf():
        try:
            subprocess.run(["perf", "record", "echo", "testing perf"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            print("Warning: perf command not found. Skipping.")
        except Exception as e:
            print(f"Warning: perf does not have permission to run. Skipping. Error: {e}")
        return False

    @staticmethod
    def can_run_flamegraph():
        if not MariaDB.can_run_perf():
            return False
        if not shutil.which("stackcollapse-perf.pl"):
            print("Warning: Command 'stackcollapse-perf.pl' missing. Skipping.")
            return False
        if not shutil.which("flamegraph.pl"):
            print("Warning: Command 'flamegraph.pl' missing. Skipping.")
            return False
        return True

    def perf_start(self, name):
        if not self._perf_record and not self._perf_stat:
            return

        # Clean up previous record if any
        self.perf_stop()

        if self._perf_record:
            record_name = f"perf.data.{name}.{self._test_time}"
            # perf record -p $(pidof mariadbd) --freq=99 --output=perf.data.{name} --timestamp-filename"
            self._perf_proc = subprocess.Popen([
                "perf",
                "record",
                "-p",
                f"{self._mariadbd_proc.pid}",
                "-g",
                "--freq=100",
                "--output=results/" + record_name,
            ], stdout=sys.stdout, stderr=sys.stderr)
            self._perf_records.append(record_name)
        elif self._perf_stat:
            stat_name = f"perf.stat.{name}.{self._test_time}"
            self._perf_proc = subprocess.Popen([
                "perf",
                "stat",
                "-x,", # split by comma
                f"--output=results/{stat_name}",
                "-p",
                f"{self._mariadbd_proc.pid}"
            ], stdout=sys.stdout, stderr=sys.stderr)
            self._perf_stats.append(stat_name)

    def perf_stop(self):
        if (self._perf_record or self._perf_stat) and self._perf_proc is not None:
            self._perf_proc.send_signal(signal.SIGINT) # perf needs to be terminated gracefully with SIGINT
            try:
                self._perf_proc.wait(10)
                print("\nPerf process terminated.")
            except subprocess.TimeoutExpired:
                print("\nError: Perf process did not terminate within the timeout period.")
            self._perf_proc = None

    def perf_analysis(self):
        if self._perf_record:
            for record in self._perf_records:
                try:
                    flamegraph_cmd = f"perf script -i results/{record} | stackcollapse-perf.pl | flamegraph.pl > results/{record}.svg"
                    subprocess.run(flamegraph_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to generate flame graph. Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        if self._perf_stat:
            for stat_file in self._perf_stats:
                try:
                    with open(f"results/{stat_file}", 'r') as file:
                        values = [int(line.split(',')[0]) for line in file if 'cpu_core/cycles/' in line or 'cpu_atom/cycles/' in line]
                        print(f"CPU cycles in {stat_file}: {sum(values):,.0f}" if values else "Error: No CPU cycle data found.")
                except (FileNotFoundError, IOError):
                    print("Error reading the perf stat file.")

    def start_db(self):
        try:
            print("\nStarting MariaDB server...")
            print(self._mariadb_start_cmd)
            self._mariadbd_proc = subprocess.Popen(self._mariadb_start_cmd, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("ERROR: Failed to start MariaDB database:", e)
            raise

        # Server is expected to start in less than 30s
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:
                raise TimeoutError("Timeout waiting for MariaDB server to start")
            try:
                if os.path.exists(self._socket_file):
                    print("\nMariaDB server started!")
                    break
            except FileNotFoundError:
                pass
            time.sleep(1)

    def fit(self, X):
        # Prepare database and table
        print("\nPreparing database and table...")
        self._cur.execute("DROP DATABASE IF EXISTS ann")
        self._cur.execute("CREATE DATABASE ann")
        self._cur.execute("USE ann")
        self._cur.execute(f"CREATE TABLE t1 (id INT PRIMARY KEY, v BLOB NOT NULL, vector INDEX (v)) ENGINE={self._engine};")

        # Insert data
        print("\nInserting data...")
        self.perf_start("inserting")
        start_time = time.time()
        for i, embedding in enumerate(X):
            self._cur.execute("INSERT INTO t1 (id, v) VALUES (%d, %s)", (i, bytes(vector_to_hex(embedding))))
        self._cur.execute("commit")
        self.perf_stop()
        print(f"\nInsert time for {X.size} records: {time.time() - start_time}")

        # Create index
        print("\nCreating index...")
        self.perf_start("indexing")
        start_time = time.time()
        if self._metric == "angular":
            pass
        elif self._metric == "euclidean":
            # The feature is being developed
            # Currently stack will be empty for indexing in perf data as nothing is executed
            #self._cur.execute("ALTER TABLE `t1` ADD VECTOR INDEX (v);")
            pass
        else:
            pass
        self.perf_stop()
        print("\nIndex creation time:", time.time() - start_time)
        self.perf_start("searching")


    def set_query_arguments(self, ef_search):
        # Set ef_search
        self._ef_search = ef_search
        self._cur.execute("SET mhnsw_limit_multiplier = %d" % ef_search)

    def query(self, v, n):
        self._cur.execute("SELECT id FROM t1 ORDER by vec_distance(v, %s) LIMIT %d", (bytes(vector_to_hex(v)), n))
        return [id for id, in self._cur.fetchall()]

    # TODO for MariaDB, get the memory usage when index is supported:
    # def get_memory_usage(self):
    #      if self._cur is None:
    #         return 0
    #      self._cur.execute("")
    #      return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"MariaDB(m={self._m}, ef_search={self._ef_search}, engine={self._engine})"

    def done(self):
        # Shutdown MariaDB server when benchmarking done
        self._cur.execute("shutdown")
        # Stop perf for searching and do final analysis
        self.perf_stop()
        self.perf_analysis()
