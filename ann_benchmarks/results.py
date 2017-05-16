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

def _listdir_filter(path, prefix):
    for f in os.listdir(path):
        if f.startswith(prefix):
            yield f[len(prefix):]

def enumerate_result_files(dataset = None, limit = None, count = None,
        distance = None, query_dataset = None, algo = None):
    def _next_part(path, element, prefix, mapping = None):
        if not os.path.isdir(path):
            return None
        elif not element:
            return map(mapping, _listdir_filter(path, prefix))
        elif not isinstance(element, list):
            return [element]
        else:
            return element

    rdir = "results/"
    counts = _next_part(rdir, count, "k=", int)
    if not counts:
        raise StopIteration

    for c in counts:
        cdir = os.path.join("results", "k=%d" % c)
        datasets = _next_part(cdir, dataset, "dataset=")
        if not datasets:
            continue

        for d in datasets:
            ddir = os.path.join(cdir, "dataset=%s" % d)
            limits = _next_part(ddir, limit, "limit=", int)
            if not limits:
                continue

            for l in limits:
                ldir = os.path.join(ddir, "limit=%d" % l)
                distances = _next_part(ldir, distance, "distance=")
                if not distances:
                    continue

                for dst in distances:
                    dstdir = os.path.join(ldir, "distance=%s" % dst)
                    query_datasets = _next_part(
                            dstdir, query_dataset, "query_dataset=")
                    if not query_dataset:
                        query_datasets.insert(0, None)

                    for q in query_datasets:
                        if q != None:
                            qdir = os.path.join(dstdir, "query_dataset=%s" % q)
                        else:
                            qdir = dstdir
                        algos = _next_part(qdir, algo, "algo=")
                        if not algos:
                            continue

                        for a in algos:
                            adir = os.path.join(qdir, "algo=%s" % a)
                            runs = _next_part(adir, None, "")

                            if not runs:
                                continue
                            else:
                                for r in runs:
                                    rpath = os.path.join(adir, r)
                                    if not (os.path.isfile(rpath) \
                                            and rpath.endswith(".json.gz")):
                                        continue
                                    descriptor = {
                                        "count": c,
                                        "dataset": d,
                                        "limit": l,
                                        "distance": dst,
                                        "query_dataset": q,
                                        "algorithm": a,
                                        "file": r
                                    }
                                    yield (descriptor, rpath)

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
