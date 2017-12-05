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


def _get_leaf_paths(path):
    if os.path.isdir(path):
        for fragment in os.listdir(path):
            for i in _get_leaf_paths(os.path.join(path, fragment)):
                yield i
    elif os.path.isfile(path) and path.endswith(".hdf5"):
        yield path


def _leaf_path_to_descriptor(path):
    directory, _ = os.path.split(path)
    parts = directory.split(os.sep)[1:]
    descriptor = {
        "file": os.path.basename(path)
    }
    for part in parts:
        try:
            name, value = part.split("=", 1)
            if name == "k":
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

def enumerate_result_files(dataset=None, count=None, algo=None):
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
               _matches(algo, desc["algorithm"])
    for path in _get_leaf_paths("results/"):
        desc = _leaf_path_to_descriptor(path)
        if _matches_all(desc):
            yield desc, path

def get_results(dataset, count):
    for d, results in get_results_with_descriptors(dataset, count):
        yield results

def get_results_with_descriptors(dataset, count):
    for d, fn in enumerate_result_files(dataset, count):
        f = h5py.File(fn)
        yield d, f
