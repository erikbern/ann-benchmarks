from __future__ import absolute_import

import os
import gzip
import json

def store_results(results, dataset, limit, count, distance, query_dataset = None):
    fragments = {
        "ds": dataset,
        "l": limit,
        "k": count,
        "dst": distance,
        "qds": query_dataset,
        "inst": results["name"],
        "algo": results["library"]
    }
    for k, v in fragments.items():
        if v and (isinstance(v, str) or isinstance(v, unicode)):
            assert not os.sep in v, """\
error: path fragment "%s" contains a path separator and so would break the \
directory hierarchy""" % k
    def _make_path(*args):
        return os.path.join(*map(lambda s: s % fragments, args))
    fn = None
    if query_dataset:
        fn = _make_path("results", "k=%(k)d", "dataset=%(ds)s",
                "limit=%(l)d", "distance=%(dst)s", "query_dataset=%(qds)s",
                "algo=%(algo)s", "%(inst)s.json.gz")
    else:
        fn = _make_path("results", "k=%(k)d", "dataset=%(ds)s",
                "limit=%(l)d", "distance=%(dst)s", "algo=%(algo)s",
                "%(inst)s.json.gz")
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    with gzip.open(fn, "w") as fp:
        fp.write(json.dumps(results) + "\n")

def _get_leaf_paths(path):
    if os.path.isdir(path):
        for fragment in os.listdir(path):
            for i in _get_leaf_paths(os.path.join(path, fragment)):
                yield i
    elif os.path.isfile(path) and path.endswith(".json.gz"):
        yield path

def _leaf_path_to_descriptor(path):
    directory, _ = os.path.split(path)
    parts = directory.split(os.sep)[1:]
    descriptor = {
        "file": os.path.basename(path),
        # This is the only thing that might not appear in the hierarchy of a
        # valid result file
        "query_dataset": None
    }
    for part in parts:
        try:
            name, value = part.split("=", 1)
            if name == "k" or name == "limit":
                value = int(value)
            # Some of the names in the hierarchy aren't the names used in the
            # descriptor; fix those up
            if name == "k":
                name = "count"
            elif name == "algo":
                name = "algorithm"
            descriptor[name] = value
        except ValueError:
            pass
    return descriptor

def enumerate_result_files(dataset = None, limit = None, count = None,
        distance = None, query_dataset = None, algo = None):
    def _matches(argv, descv):
        if argv == None:
            return True
        elif not isinstance(argv, list):
            return descv == argv
        else:
            return descv in argv
    def _matches_all(desc):
        return _matches(count, desc["count"]) and \
               _matches(dataset, desc["dataset"]) and \
               _matches(limit, desc["limit"]) and \
               _matches(distance, desc["distance"]) and \
               _matches(query_dataset, desc["query_dataset"]) and \
               _matches(algo, desc["algorithm"])
    for path in _get_leaf_paths("results/"):
        desc = _leaf_path_to_descriptor(path)
        if _matches_all(desc):
            yield desc, path

def get_results(dataset, limit, count, distance, query_dataset = None):
    for d, results in get_results_with_descriptors(
            dataset, limit, count, distance, query_dataset):
        if d["query_dataset"] == query_dataset:
            yield results

def get_results_with_descriptors(
        dataset, limit, count, distance, query_dataset):
    for d, fn in enumerate_result_files(dataset, limit, count, distance,
            query_dataset):
        with gzip.open(fn, "r") as fp:
            try:
                yield (d, json.load(fp))
            except ValueError:
                print """\
warning: loading results file %s failed, skipping""" % fn
                continue
