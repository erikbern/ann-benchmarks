from __future__ import absolute_import

import os
import json

def store_results(results, dataset, limit, count, distance, query_dataset = None):
    fn = None
    if limit == -1:
        fn = "results/%s.txt" % dataset
    else:
        fn = "results/%s-%d.txt" % (dataset, limit)
    with open(fn, "a") as fp:
        fp.write(json.dumps(results) + "\n")

def enumerate_result_files(dataset = None, limit = -1, count = None,
        distance = None, query_dataset = None):
    if dataset:
        if limit == -1:
            p = "results/%s.txt" % dataset
            if os.path.isfile(p):
                yield p
        else:
            p = "results/%s-%d.txt" % (dataset, limit)
            if os.path.isfile(p):
                yield p

def get_results(dataset, limit, count, distance, query_dataset = None):
    for fn in enumerate_result_files(dataset, limit, count, distance,
            query_dataset):
        with open(fn, "r") as fp:
            for line in fp:
                yield json.loads(line)
