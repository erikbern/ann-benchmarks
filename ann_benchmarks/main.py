from __future__ import absolute_import
import gzip, numpy, time, os, multiprocessing, argparse, pickle, resource, random, math, yaml
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import json
import shutil

from ann_benchmarks.algorithms.itu import ITUHashing, ITUFilteringDouble
from ann_benchmarks.algorithms.lshf import LSHF
from ann_benchmarks.algorithms.annoy import Annoy
from ann_benchmarks.algorithms.flann import FLANN
from ann_benchmarks.algorithms.panns import PANNS
from ann_benchmarks.algorithms.kdtree import KDTree
from ann_benchmarks.algorithms.kgraph import KGraph
from ann_benchmarks.algorithms.nearpy import NearPy
from ann_benchmarks.algorithms.nmslib import NmslibNewIndex, NmslibReuseIndex
from ann_benchmarks.algorithms.falconn import FALCONN
from ann_benchmarks.algorithms.balltree import BallTree
from ann_benchmarks.algorithms.rpforest import RPForest
from ann_benchmarks.algorithms.bruteforce import BruteForce, BruteForceBLAS
from ann_benchmarks.algorithms.subprocess import Subprocess, BitSubprocess

from ann_benchmarks.algorithms.base import BaseANN

from ann_benchmarks.data import type_info
from ann_benchmarks.distance import metrics as pd
from ann_benchmarks.constants import INDEX_DIR

def get_dataset(which='glove', limit=-1, random_state=3, test_size=10000):
    cache = 'queries/%s-%d-%d-%d.npz' % (which, test_size, limit, random_state)
    if os.path.exists(cache):
        v = numpy.load(cache)
        X_train = v['train']
        X_test = v['test']
        manifest = v['manifest'][0]
        print(manifest, X_train.shape, X_test.shape)
        return manifest, X_train, X_test
    local_fn = os.path.join('install', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    manifest = {
      'point_type': 'float'
    }
    if os.path.exists(local_fn + '.yaml'):
        y = yaml.load(open(local_fn + '.yaml'))
        if 'dataset' in y:
            manifest.update(y['dataset'])

    point_type = manifest['point_type']

    assert point_type in type_info, """\
dataset %s: unknown point type '%s'""" % (which, point_type)
    info = type_info[point_type]

    assert "parse_entry" in info, """\
dataset %s: no parser for points of type '%s'""" % (which, point_type)
    loader = info["parse_entry"]

    X = []
    for i, line in enumerate(f):
        X.append(loader(line))
        if limit != -1 and len(X) == limit:
            break
    X = numpy.array(X, dtype=info.get("type"))

    if "finish_entries" in info:
        X = info["finish_entries"](X)

    import sklearn.cross_validation

    # Here Erik is most welcome to use any other random_state
    # However, it is best to use a new random seed for each major re-evaluation,
    # so that we test on a trully bind data.
    X_train, X_test = \
      sklearn.cross_validation.train_test_split(
          X, test_size=test_size, random_state=random_state)
    print(X_train.shape, X_test.shape)

    numpy.savez(cache, manifest=[manifest], train=X_train, test=X_test)
    return manifest, X_train, X_test

def run_algo(X_train, queries, library, algo, distance, results_fn):
    try:
        t0 = time.time()
        if algo != 'bf':
            algo.fit(X_train)
        build_time = time.time() - t0
        print('Built index in', build_time)

        best_search_time = float('inf')
        best_precision = 0.0 # should be deterministic but paranoid
        for i in xrange(3): # Do multiple times to warm up page cache, use fastest
            def single_query(t):
                v, _, _ = t
                start = time.time()
                found = algo.query(v, 10)
                total = (time.time() - start)
                found = map(
                    lambda idx: (int(idx), float(pd[distance](v, X_train[idx]))),
                    list(found))
                return (total, found)
            if algo.use_threads():
                pool = multiprocessing.pool.ThreadPool()
                results = pool.map(single_query, queries)
            else:
                results = map(single_query, queries)

            total_time = sum(map(lambda (time, _): time, results))
            search_time = total_time / len(queries)
            best_search_time = min(best_search_time, search_time)

        output = {
            "library": library,
            "name": algo.name,
            "build_time": build_time,
            "best_search_time": best_search_time,
            "results": results
        }
        print(output)

        f = open(results_fn, 'a')
        f.write(json.dumps(output) + "\n")
        f.close()
    finally:
        algo.done()

def compute_distances(distance, X_train, X_test):
    print('computing max distances for queries...')

    bf = BruteForceBLAS(distance, precision=X_train.dtype)
    # Prepare queries
    bf.fit(X_train)
    queries = []
    for x in X_test:
        correct = bf.query_with_distances(x, 10)
        max_distance = max(correct, key=lambda (_, distance): distance)[1]
        queries.append((x, max_distance, correct))
        if len(queries) % 100 == 0:
            print(len(queries), '...')

    return queries

def get_fn(base, args):
    fn = os.path.join(base, args.dataset)

    if args.limit != -1:
        fn += '-%d' % args.limit
    if os.path.exists(fn + '.gz'):
        fn += '.gz'
    else:
        fn += '.txt'

    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)

    return fn

from itertools import product

with open("algos.yaml", "r") as f:
    _algorithms = yaml.load(f)

def handle_args(args):
    if isinstance(args, list):
        args = map(lambda el: el if isinstance(el, list) else [el], args)
        return map(list, product(*args))
    elif isinstance(args, dict):
        flat = []
        for k, v in args.iteritems():
            if isinstance(v, list):
                flat.append(map(lambda el: (k, el), v))
            else:
                flat.append([(k, v)])
        return map(dict, product(*flat))
    else:
        raise TypeError("No args handling exists for %s" % type(args).__name__)

point_type = "float"
distance_metric = "euclidean"

def get_definitions(point_type, distance_metric):
    algorithm_definitions = {}
    if "any" in _algorithms[point_type]:
        algorithm_definitions.update(_algorithms[point_type]["any"])
    algorithm_definitions.update(_algorithms[point_type][distance_metric])

    algos = {}
    for (name, algo) in algorithm_definitions.iteritems():
        assert "constructor" in algo, \
                "group %s does not specify a constructor" % name
        cn = algo["constructor"]
        assert cn in globals(), \
                "group %s specifies the nonexistent constructor %s" % (name, cn)
        constructor = globals()[cn]

        algos[name] = []

        base_args = []
        vs = {
            "@metric": distance_metric
        }
        if "base-args" in algo:
            base_args = map(lambda arg: arg \
                            if not isinstance(arg, str) or \
                            not arg in vs else vs[arg],
                            algo["base-args"])

        for run_group in algo["run-groups"].values():
            if "arg-groups" in run_group:
                groups = []
                for arg_group in run_group["arg-groups"]:
                    if isinstance(arg_group, dict):
                        # Dictionaries need to be expanded into lists in order
                        # for the subsequent call to handle_args to do the
                        # right thing
                        groups.append(handle_args(arg_group))
                    else:
                        groups.append(arg_group)
                args = handle_args(groups)
            elif "args" in run_group:
                args = handle_args(run_group["args"])
            else:
                assert False, "? what? %s" % run_group

            for arg_group in args:
                obj = None
                try:
                    aargs = []
                    aargs.extend(base_args)
                    if isinstance(arg_group, list):
                        aargs.extend(arg_group)
                    else:
                        aargs.append(arg_group)
                    obj = constructor(*aargs)
                    algos[name].append(obj)
                except Exception as e:
                    print e
                    pass
    return algos

def main():
    # Set resource limits to prevent memory bombs
    memory_limit = 12 * 2**30
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if soft == resource.RLIM_INFINITY or soft >= memory_limit:
        print('resetting memory limit from', soft, 'to', memory_limit)
        resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))

    # Nmslib specific code
    # Remove old indices stored on disk
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Which dataset', default='glove')
    parser.add_argument('--distance', help='Distance', default='angular')
    parser.add_argument('--limit', help='Limit', type=int, default=-1)
    parser.add_argument('--algo', help='run only this algorithm', default=None)
    parser.add_argument('--sub-algo', help='run only the named algorithm instance', default=None)
    parser.add_argument('--no_save_index', help='Do not save indices', action='store_true')

    args = parser.parse_args()

    manifest, X_train, X_test = get_dataset(args.dataset, args.limit)

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)

    print('storing queries in', queries_fn, 'and results in', results_fn)

    if not os.path.exists(queries_fn):
        queries = compute_distances(args.distance, X_train, X_test)
        f = open(queries_fn, 'w')
        pickle.dump(queries, f)
        f.close()
    else:
        queries = pickle.load(open(queries_fn))

    print('got', len(queries), 'queries')

    algos_already_ran = set()
    if os.path.exists(results_fn):
        for line in open(results_fn):
            run = json.loads(line)
            algos_already_ran.add((run["library"], run["name"]))

    point_type = manifest['point_type']
    algos = get_definitions(point_type, args.distance)

    if args.algo:
        print('running only', args.algo)
        algos = {args.algo: algos[args.algo]}
        if args.sub_algo:
            algos[args.algo] = \
              [algo for algo in algos[args.algo] if algo.name == args.sub_algo]

    algos_flat = []

    for library in algos.keys():
        for algo in algos[library]:
            if (library, algo.name) not in algos_already_ran:
                algos_flat.append((library, algo))

    random.shuffle(algos_flat)

    print('order:', [a.name for l, a in algos_flat])

    for library, algo in algos_flat:
        print(algo.name, '...')
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(
            target=run_algo,
            args=(X_train, queries, library, algo, args.distance, results_fn))
        p.start()
        p.join()

if __name__ == '__main__':
    main()
