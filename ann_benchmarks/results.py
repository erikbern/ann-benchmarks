from __future__ import absolute_import

import h5py
import os
from ann_benchmarks.algorithms.definitions import get_result_filename


def store_results(dataset, count, definition, attrs, results):
    fn = get_result_filename(dataset, count, definition)
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(fn, 'w')
    for k, v in attrs.items():
        f.attrs[k] = v
    times = f.create_dataset('times', (len(results),), 'f')
    neighbors = f.create_dataset('neighbors', (len(results), count), 'i')
    distances = f.create_dataset('distances', (len(results), count), 'f')
    for i, (time, ds) in enumerate(results):
        times[i] = time
        neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
        distances[i] = [d for n, d in ds] + [float('inf')] * (count - len(ds))
    f.close()


def load_results(dataset, count, definitions):
    for definition in definitions:
        fn = get_result_filename(dataset, count, definition)
        if os.path.exists(fn):
            f = h5py.File(fn)
            yield definition, f
            f.close()

def _get_leaf_paths(path):
    if os.path.isdir(path):
        for fragment in os.listdir(path):
            for i in _get_leaf_paths(os.path.join(path, fragment)):
                yield i
    elif os.path.isfile(path):
        yield path

def load_all_results():
    import re
    #TODO(Martin) Must be adaptive to get_result_filename
    regex = r"results\/(?P<dataset>[^\/]+)\/(?P<count>\d+)\/(?P<algorithm>[^\/]+)\/(?P<distance>[^_]+)_(?P<params>.*)"
    for fn in _get_leaf_paths("results/"):
        match = re.match(regex, fn)
        if match:
            d = {}
            for k in ["count", "dataset", "distance", "algorithm"]:
                d[k] = match.group(k)
            d["count"] = int(d["count"])
            f = h5py.File(fn)
            yield d, f
            f.close()

