import subprocess
import sys
import requests

from ..base.module import BaseANN
from time import sleep

class SurrealMtree(BaseANN):        

    def __init__(self, metric, path = 'memory', capacity = 40):
        self._metric = metric
        self._path = path
        self._capacity = capacity
        subprocess.run(f"surreal start --allow-all -u ann -p ann -b 127.0.0.1:8000 {path}  &", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("wait for the server to be up...")
        sleep(5)
        self._session = requests.Session()
        self._session.auth = ('ann', 'ann')
        headers={
            "NS": 'ann',
            "DB": 'ann',
            "Accept": "application/json",
        }
        self._session.headers.update(headers)

    def _sql(self, q):
        r = self._session.post(f'http://127.0.0.1:8000/sql', q)
        if r.status_code != 200:
            raise RuntimeError(f"{r}")
        return r

    def fit(self, X):
        print("copying data...")

        dim = X.shape[1]

        # Fit the database per batch
        batch = max(10000 // dim, 1)
        q = ""
        l = 0
        t = 0
        for i, embedding in enumerate(X):
            l += 1
            q += f"CREATE items:{i} SET i={i}, r={embedding.tolist()};"
            if l == batch:
                self._sql(q)
                q = ''
                t += l
                l = 0
        if l > 0:
            self._sql(q)
            t += l
        print(f"{t} vectors ingested")

        print(f"creating index - capacity: {self._capacity} - dim: {dim}")
        if self._metric == "euclidean":
            self._sql(f"DEFINE INDEX ix ON items FIELDS r MTREE DIMENSION {dim} DIST EUCLIDEAN TYPE F64 CAPACITY {self._capacity};")
        else:
            raise RuntimeError(f"unknown metric {self.metric}")
        print("Index construction done")     

    def query(self, v, n):
        r = self._sql(f"SELECT i FROM items WHERE r <{n}> {v.tolist()};")
        j = r.json()
        c = 0
        items = []
        for item in j[0]['result']:
            c += 1
            items.append(item['i'])
        if c != n:
            raise RuntimeError(f"Invalid items count: {c} => {j}")
        return items

    def __str__(self):
        return f"SurrealMtree(path={self._path}, capacity={self._capacity})"

    def close(self):
        self._session.close()