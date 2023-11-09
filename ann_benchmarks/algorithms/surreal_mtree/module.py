import subprocess
import sys
import json

from ..base.module import BaseANN
from surrealdb import SurrealDB
from time import sleep

class SurrealMtree(BaseANN):        

    def __init__(self, metric, path = 'memory', capacity = 40):
        self._metric = metric
        self._path = path
        self._capacity = capacity
        subprocess.run(f"surreal start --allow-all -u ann -p ann -b 127.0.0.1:8000 {path}  &", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("wait for the server to be up...")
        sleep(5)
        self._connection = SurrealDB("ws://127.0.0.1:8000/ann/ann")
        self._connection.signin({
            "username": "ann",
            "password": "ann",
        })  

    def _query(self, q):
        r = self._connection.query(q)
        if r is None:
            raise RuntimeError(f"None response: {q}")
        if len(r) != 1:
            raise RuntimeError(f"Incorrect response: {q} => {r}")
        return r

    def fit(self, X):
        print("copying data...")
        for i, embedding in enumerate(X):
            self._query(f"CREATE items:{i} SET i={i}, r={embedding.tolist()};")
        dim = X.shape[1]
        print("creating index...")
        if self._metric == "euclidean":
            self._query(f"DEFINE INDEX ix ON items FIELDS r MTREE DIMENSION {dim} DIST EUCLIDEAN TYPE F64 CAPACITY {self._capacity};")
        else:
            raise RuntimeError(f"unknown metric {self.metric}")
        print("Index construction done")     

    def query(self, v, n):
        # q = f"SELECT i FROM items WHERE r <{n}> {v.tolist()};"
        self._query('SELECT count() FROM items;')
        res = self._connection.query(q)
        return [item['i'] for item in res[0]['result']]

    def __str__(self):
        return f"SurrealMtree(path={self._path}, capacity={self._capacity})"