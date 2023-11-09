import subprocess
import sys
import asyncio
import threading

from ..base.module import BaseANN
from surrealdb import Surreal
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from functools import partial

class SurrealMtree(BaseANN):        

    def __init__(self, metric, path = 'memory', capacity = 40):
        self.loop = asyncio.get_event_loop()
        self.metric = metric
        self.path = path
        self.capacity = capacity
        subprocess.run(f"surreal start --allow-all -u ann -p ann -b 127.0.0.1:8000 {path}  &", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("wait for the server to be up...")
        sleep(5)

    async def async_connect(self, db):
        await db.connect()
        await db.signin({"user": "ann", "pass": "ann"})
        await db.use("ann", "ann")

    async def async_fit(self, X):
        async with Surreal("ws://localhost:8000/rpc") as db:
            await self.async_connect(db)
            print("copying data...")
            for i, embedding in enumerate(X):
                await db.create(
                    f"items:{i}",
                    { "i": i, "r": embedding.tolist()},
                )
            dim = X.shape[1];
            print("creating index...")
            if self.metric == "euclidean":
                await db.query(f"DEFINE INDEX ix ON items FIELDS r MTREE DIMENSION {dim} DIST EUCLIDEAN TYPE F64 CAPACITY {self.capacity}")
            else:
                raise RuntimeError(f"unknown metric {self.metric}")
            print("Index construction done")

    def fit(self, X):
        self.loop.run_until_complete(self.async_fit(X))

    async def async_query(self, v, n):
         async with Surreal("ws://localhost:8000/rpc") as db:
            await self.async_connect(db)
            res = await db.query(f"SELECT i FROM items WHERE r <{n}> {v.tolist()}")
            return [item['i'] for item in res[0]['result']]

    def query(self, v, n):
        return self.loop.run_until_complete(self.async_query(v, n))

    def __str__(self):
        return f"SurrealMtree(path={self.path}, capacity={self.capacity})"