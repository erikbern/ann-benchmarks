import subprocess
import sys
import requests

from ..base.module import BaseANN
from time import sleep

class SurrealMtree(BaseANN):        

    def __init__(self, metric, path = 'memory', capacity = 40, doc_ids_cache = 100, mtree_cache = 100):
        self._metric = metric
        self._path = path
        self._capacity = capacity
        self._doc_ids_cache = doc_ids_cache
        self._mtree_cache = mtree_cache
        subprocess.run(f"surreal start --allow-all -u ann -p ann -b 127.0.0.1:8000 {path}  &", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("wait for the server to be up...")
        sleep(5)
        self._session = requests.Session()
        self._session.auth = ('ann', 'ann')
        headers={
            "surreal-ns": 'ann',
            "surreal-db": 'ann',
            "Accept": "application/json",
        }
        self._session.headers.update(headers)

    def _sql(self, q):
        r = self._session.post('http://127.0.0.1:8000/sql', q)
        if r.status_code != 200:
            raise RuntimeError(f"{r.text}")
        return r

    def _create_index(self, dim):
        if self._metric == "euclidean":
            dist = 'EUCLIDEAN'
        elif self._metric == 'manhattan':
            dist = 'MANHATTAN'
        else:
            raise RuntimeError(f"unknown metric {self.metric}")
        sql = "REMOVE INDEX IF EXISTS ix ON items;\nREMOVE TABLE IF EXISTS items;\n"
        sql += f"DEFINE INDEX ix ON items FIELDS r MTREE DIMENSION {dim} DIST {dist} TYPE F32 CAPACITY {self._capacity} DOC_IDS_CACHE {self._doc_ids_cache} MTREE_CACHE {self._mtree_cache};"
        print(f"\r{sql}")
        self._checked_sql(sql)


    def _ingest(self, dim, X): 
        # Fit the database per batch
        print("Ingesting vectors...")
        batch = max(20000 // dim, 1)
        q = ""
        l = 0
        t = 0
        for i, embedding in enumerate(X):
            v = embedding.tolist()
            l += 1
            q += f"CREATE items:{i} SET r={v} RETURN NONE;"
            if l == batch:
                self._checked_sql(q)
                q = ''
                t += l
                l = 0
                print(f"\r{t} vectors ingested", end = '')
        if l > 0:
            self._checked_sql(q)
            t += l
            print(f"\r{t} vectors ingested", end = '')

    def fit(self, X):
        dim = X.shape[1]
        self._create_index(dim)
        self._ingest(dim, X)
        print("\nIndex construction done")     

    def _checked_sql(self, q):
        res = self._sql(q).json()
        for r in res:
            if r['status'] != 'OK':
                raise RuntimeError(f"Error: {r}")
        return res
            
    def query(self, v, n):
        v = v.tolist()
        j = self._checked_sql(f"SELECT id FROM items WHERE r <|{n}|> {v};")
        items = []
        for item in j[0]['result']:
            id = item['id']
            items.append(int(id[6:]))
        return items

    def __str__(self):
        return f"SurrealMtree(path={self._path}, capacity={self._capacity}, doc_ids_cache={self._doc_ids_cache}, mtree_cache={self._mtree_cache})"

    def done(self) -> None:
        self._session.close()
        subprocess.run("pkill surreal", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
