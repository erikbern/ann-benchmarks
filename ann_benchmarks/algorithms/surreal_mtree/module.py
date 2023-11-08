import subprocess
import sys
import asyncio

from ..base.module import BaseANN
from surrealdb import Surreal
from concurrent.futures import ThreadPoolExecutor
from time import sleep

class SurrealMtree(BaseANN):
    def __init__(self, metric, method_param):
        self.executor = ThreadPoolExecutor()
        self.metric = metric
        self.ef = method_param['efConstruction']
        self.m = method_param["M"]
        self.db = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s::real[] LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s::real[] LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    async def async_fit(self, X):
        subprocess.run("surreal start --allow-all -u ann -p ann -b 127.0.0.1:8000 memory  &", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("wait for the server to be up...")
        sleep(5);
        self.db = Surreal("ws://127.0.0.1:8000/rpc")
        await self.db.connect()
        await self.db.signin({"user": "ann", "pass": "ann"})
        await self.db.use("ann", "ann")
        print("copying data...")
        for i, embedding in enumerate(X):
            await self.db.create(
                f"items:{id}",
                { "real": embedding.tolist()},
            )
        print("creating index...%d", X.shape[1])
        if self.metric == "euclidean":
            print(await self.db.query("DEFINE INDEX ix ON items FIELDS real MTREE DIMENSION %d DIST EUCLIDEAN TYPE F64" % X.shape[1]))
        else:
            raise RuntimeError(f"unknown metric {self.metric}")
        print("Index construction done!")

    def fit(self, X):
        future = self.executor.submit(asyncio.run, self.async_fit(X))
        future.result()

    def set_query_arguments(self, ef):
        self.ef = ef

    async def async_query(self, v, n):
        q = f"SELECT id FROM items WHERE real <{n}> {v.tolist()}"
        print(q)
        r = await self.db.query(q)
        print(r)
        return [id for id, in self._cur.fetchall()]

    def query(self, v, n):
        future = self.executor.submit(asyncio.run, self.async_query(v, n))
        return future.result()

    def __str__(self):
        return f"SurrealMtree(m={self.m}, ef={self.ef})"

    def close(self):
        self.executor.shutdown(wait=True)
