# Implementation of pgANN from https://github.com/netrasys/pgANN
# Stores reduced embeddings (Umap) in PostGres Cube structure 
# for nearest neighour search. Smaller Umap embeddings allow for 
# faster searches. On-disk (out-of-core) so will be slower than 
# in-RAM methods.
# NOTE: Having problems fitting the whole Glove-100 training set
# with Umap. Either sampling or randomly initializing the Umap 
# training. This adds more sources of error and should be fixed 
# before any actual benchmarks.  
from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
import numpy as np
import umap
import subprocess
import psycopg2
import dataset

class PgANN(BaseANN):
    def __init__(self, metric, embed_size=None, sample_prc=None, num_neigh=12):
        self.metric = metric
        if metric == 'angular':
            self.metric = 'cosine'
        self.embed_size = embed_size
        self.sample_prc = sample_prc
        self.num_neigh = num_neigh
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
        # create GIST
        gist_creation = """
        CREATE INDEX ix_gist 
        ON ann USING GIST (embeddings);
        """
        self.cursor.execute(gist_creation)


    def fit(self, X):
        # reduce vector dimension
        if self.embed_size:
            # sample or initiate random Umap fit
            if self.sample_prc:
                # UMAP errors out with really big data.
                # Can either sample or initiate
                # randomly (instead of a spectral embeddings)
                if self.sample_prc <= 0:
                    # no sample, random initialization
                    init = 'random'
                    X_train = X
                else:
                    # sample, spectral embeddings
                    init = 'spectral'
                    rand = np.random.choice(
                        np.arange(len(X)), 
                        int(len(X)*self.sample_prc), 
                        replace=False)
                    X_train = X[rand]
            # no sample, use spectral embeddings
            else:
                init = 'spectral'
                X_train = X 

            self.reducer = umap.UMAP(
                metric = self.metric,
                n_neighbors = self.num_neigh, 
                n_components = self.embed_size,
                init = init).fit(X_train)
            embeddings = self.reducer.transform(X)
        # no vector reduction, really slow querying
        else:
            embeddings = X

        # populate DB
        for t, embed in enumerate(embeddings):
            embed_string= "({0})".format(
                ','.join("%10.8f" % x for x in embed))
            insert="""
            INSERT INTO ann (index_num, embeddings) 
            VALUES ({}, Cube'{}');
            """.format(t, embed_string)
            self.cursor.execute(insert)


    def set_query_arguments(self, thresh):
        self.thresh = thresh


    def query(self, v, n):
        # transfrom query vector
        if self.embed_size:
            v = self.reducer.transform([v])[0]
        # convert to string
        emb_string = "'({0})'".format(','.join("%10.8f" % x for x in v))
        # create query
        query = """
        SELECT index_num from ann 
        WHERE embeddings <-> cube({}) < {}
        ORDER BY embeddings <-> cube({}) ASC
        LIMIT {} 
        """.format(emb_string, self.thresh, emb_string, n)
        # run and get query results
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return [r[0] for r in results]


    def __str__(self):
        return ('PgANN(threshold={}, embed_size={}, \
            sample_prc={}, num_neigh={}').format(
            self.thresh, self.embed_size,
            self.sample_prc, self.num_neigh) 
