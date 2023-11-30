import subprocess
import sys
import requests

from ..base.module import BaseANN
from time import sleep

class SurrealBruteForce(BaseANN):        

    def __init__(self, metric, path = 'memory', parallel = ''):
        if metric == "euclidean":
            self._metric = 'EUCLIDEAN'
        elif metric == 'manhattan':
            self._metric = 'MANHATTAN'
        elif metric == 'angular':
            self._metric = 'COSINE'
        elif metric == 'hamming':
            self._metric = 'HAMMING'
        elif metric == 'jaccard':
            self._metric = 'JACCARD'
        else:
            raise RuntimeError(f"unknown metric {metric}")
        self._path = path
        self._parallel = parallel
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
        r = self._session.post('http://127.0.0.1:8000/sql', q)
        if r.status_code != 200:
            raise RuntimeError(f"{r.text}")
        return r

    def _ingest(self, dim, X): 
        # Fit the database per batch
        print("Ingesting vectors...")
        batch = max(20000 // dim, 1)
        q = ""
        l = 0
        t = 0
        for i, embedding in enumerate(X):
            v = embedding.tolist();
            l += 1
            q += f"CREATE items:{i} SET r={v};"
            if l == batch:
                self._checked_sql(q)
                q = ''
                t += l
                l = 0
                print(f"{t} vectors ingested")
        if l > 0:
            self._checked_sql(q)
            t += l
        print(f"{t} vectors ingested")

    def fit(self, X):
        dim = X.shape[1];
        self._ingest(dim, X)
        print("Index construction done")     

    def _checked_sql(self, q):
        res = self._sql(q).json()
        for r in res:
            if r['status'] != 'OK':
                raise RuntimeError(f"Error: {r}")
        return res
            
    def query(self, v, n):
        v = v.tolist();
        j = self._checked_sql(f"SELECT id FROM items WHERE r <{n},{self._metric}> {v} {self._parallel};")
        c = 0
        items = []
        for item in j[0]['result']:
            c += 1
            items.append(int(item['id'][6:]))
        if c != n:
            raise RuntimeError(f"Invalid items count: {c} => {j}")
        return items

    def __str__(self):
        return f"SurrealBruteForce(path={self._path}, parallel={self._parallel})"

    def done(self) -> None:
        self._session.close()
        subprocess.run("pkill surreal", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
