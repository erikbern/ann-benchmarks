# Implementation of pgANN from https://github.com/netrasys/pgANN
# Stores vectors as GIST index cube structures for nearest neighour 
# search in postgresql. Can possibly reduce vector with Umap.
# Slow for benchmark, ideally want separate postgresql server.
# A good baseline for on-disk nearest neighbour search.
from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
import numpy as np
import umap
import subprocess
import psycopg2

class PgANN(BaseANN):
    def __init__(self, metric, embed_size=None, num_neigh=12, 
                 init='random'):
        self.metric = metric
        if metric == 'angular':
            self.metric = 'cosine'
        self.embed_size = embed_size
        if embed_size == 'None':
            self.embed_size = None
        self.num_neigh = num_neigh
        self.init = init
        # restart cluster (remove stale pid)
        args = ['pg_ctlcluster', '10', 'main', 'restart']
        subprocess.call(args)
        # connect to DB
        conn = psycopg2.connect(user="postgres", host="localhost", 
                                password="password", port="5432", 
                                database="ann_benchmark")
        self.cursor = conn.cursor()
        # create cube extension
        self.cursor.execute("CREATE EXTENSION cube;")
        # create table
        table_creation = """
            CREATE TABLE ann (
                id serial PRIMARY KEY, 
                index_num integer, 
                embeddings cube);
        """
        self.cursor.execute(table_creation)
        # create GIST index on embeddings
        gist_creation = """
            CREATE INDEX ix_gist 
            ON ann USING GIST (embeddings);
        """
        self.cursor.execute(gist_creation)


    def fit(self, X):
        if self.embed_size:
            # reduce vector dimension
            self.reducer = umap.UMAP(
                metric = self.metric,
                n_neighbors = self.num_neigh, 
                n_components = self.embed_size,
                init = self.init).fit(X)
            embeddings = self.reducer.transform(X)
        else:
            # no reduction
            embeddings = X
        # populate DB
        for t, embed in enumerate(embeddings):
            embed_string= "'({0})'".format(
                ','.join("%10.8f" % x for x in embed))
            insert="""
                INSERT INTO ann (index_num, embeddings) 
                VALUES ({0}, cube{1});
            """.format(t, embed_string)
            self.cursor.execute(insert)


    def set_query_arguments(self, thresh):
        self.thresh = thresh


    def query(self, v, n):
        # transfrom query vector
        if self.embed_size:
            v = self.reducer.transform([v])[0]
        # convert to string
        emb_string = "'({0})'".format(','.join(
            "%10.8f" % x for x in v))
        # create query
        query = """
            SELECT index_num FROM ann 
            WHERE embeddings <-> cube({0}) < {1}
            ORDER BY embeddings <-> cube({0}) ASC
            LIMIT {2};
        """.format(emb_string, self.thresh, n)
        # run and get query results
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return [r[0] for r in results]


    def __str__(self):
        return ('PgANN(threshold={}, embed_size={}, \
            num_neigh={}, init={}').format(
            self.thresh, self.embed_size,
            self.num_neigh, self.init) 

