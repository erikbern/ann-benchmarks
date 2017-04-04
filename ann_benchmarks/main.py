from __future__ import absolute_import
import time, os, multiprocessing, argparse, pickle, resource, random, math
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import sys
import json
import shutil
import importlib
import traceback

from ann_benchmarks.datasets import get_dataset, split_dataset, get_query_cache_path
from ann_benchmarks.distance import metrics as pd
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
from ann_benchmarks.algorithms.definitions import get_algorithms, get_definitions

def run_algo(X_train, queries, library, algo, distance, results_fn,
        run_count=3, force_single=False):
    try:
        prepared_queries = False
        if hasattr(algo, "supports_prepared_queries"):
            prepared_queries = algo.supports_prepared_queries()

        t0 = time.time()
        if algo != 'bf':
            index_size_before = algo.get_index_size("self")
            algo.fit(X_train)
            build_time = time.time() - t0
            index_size = algo.get_index_size("self") - index_size_before
            print('Built index in', build_time)
            print('Index size: ', index_size)

        best_search_time = float('inf')
        for i in xrange(run_count):
            def single_query(t):
                v, _, _ = t
                if prepared_queries:
                    algo.prepare_query(v, 10)
                    start = time.time()
                    algo.run_prepared_query()
                    total = (time.time() - start)
                    candidates = algo.get_prepared_query_results()
		    print candidates
                else:
                    start = time.time()
                    candidates = algo.query(v, 10)
                    total = (time.time() - start)
                candidates = map(
                    lambda idx: (int(idx), float(pd[distance](v, X_train[idx]))),
                    list(candidates))
                return (total, candidates)
            if algo.use_threads() and not force_single:
                pool = multiprocessing.pool.ThreadPool()
                results = pool.map(single_query, queries)
            else:
                results = map(single_query, queries)

            total_time = sum(map(lambda (time, _): time, results))
            total_candidates = sum(map(lambda (_, candidates): len(candidates), results))
            search_time = total_time / len(queries)
	    avg_candidates = total_candidates / len(queries)
            best_search_time = min(best_search_time, search_time)

        output = {
            "library": library,
            "name": algo.name,
            "build_time": build_time,
            "best_search_time": best_search_time,
            "index_size": index_size,
            "results": results,
            "candidates": avg_candidates,
            "run_count": run_count,
            "run_alone": force_single
        }

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

def get_fn(base, dataset, limit=-1):
    fn = os.path.join(base, dataset)

    if limit != -1:
        fn += '-%d' % limit
    if os.path.exists(fn + '.gz'):
        fn += '.gz'
    else:
        fn += '.txt'

    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)

    return fn

def positive_int(s):
    i = None
    try:
        i = int(s)
    except ValueError:
        pass
    if not i or i < 1:
        raise argparse.ArgumentTypeError("%r is not a positive integer" % s)
    return i

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--dataset',
            metavar='NAME',
            help='the dataset to load training points from',
            default='glove')
    parser.add_argument(
            '--query-dataset',
            metavar='NAME',
            help='load query points from another dataset instead of choosing them randomly from the training dataset',
            default=None)
    parser.add_argument(
            '--distance',
            help='the metric used to calculate the distance between points',
            default='angular')
    parser.add_argument(
            '--limit',
            help='the maximum number of points to load from the dataset, or -1 to load all of them',
            type=int,
            default=-1)
    parser.add_argument(
            '--definitions',
            metavar='FILE',
            help='load algorithm definitions from FILE',
            default='algos.yaml')
    parser.add_argument(
            '--algorithm',
            metavar='NAME',
            help='run only the named algorithm',
            default=None)
    parser.add_argument(
            '--sub-algorithm',
            metavar='NAME',
            help='run only the named instance of an algorithm (requires --algo)',
            default=None)
    parser.add_argument(
            '--list-algorithms',
            help='print the names of all known algorithms and exit',
            action='store_true',
            default=argparse.SUPPRESS)
    parser.add_argument(
            '--force',
            help='''\
run algorithms even if results for them already exist (note that this option \
will produce results files with duplicate entries)''',
            action='store_true')
    parser.add_argument(
            '--runs',
            metavar='COUNT',
            type=positive_int,
            help='run each algorithm instance %(metavar)s times and use only the best result',
            default=3)
    parser.add_argument(
            '--single',
            help='run only a single algorithm instance at a time',
            action='store_true')
    parser.add_argument(
            '--no_save_index',
            help='do not save indices',
            action='store_true')

    args = parser.parse_args()

    definitions = get_definitions(args.definitions)
    if hasattr(args, "list_algorithms"):
        print "The following algorithms are supported..."
        for point in definitions:
            print "\t... for the point type '%s'..." % point
            for metric in definitions[point]:
                print "\t\t... and the distance metric '%s':" % metric
                for algorithm in definitions[point][metric]:
                    print "\t\t\t%s" % algorithm
        sys.exit(0)

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

    # Note that algos.yaml references this; it should be somewhere more
    # permanent
    constructors_ = [
      ("ann_benchmarks.algorithms.itu",
          ["ITUHashing", "ITUFilteringDouble"]),
      ("ann_benchmarks.algorithms.lshf",
          ["LSHF"]),
      ("ann_benchmarks.algorithms.annoy",
          ["Annoy"]),
      ("ann_benchmarks.algorithms.flann",
          ["FLANN"]),
      ("ann_benchmarks.algorithms.panns",
          ["PANNS"]),
      ("ann_benchmarks.algorithms.kdtree",
          ["KDTree"]),
      ("ann_benchmarks.algorithms.kgraph",
          ["KGraph"]),
      ("ann_benchmarks.algorithms.nearpy",
          ["NearPy"]),
      ("ann_benchmarks.algorithms.nmslib",
          ["NmslibNewIndex", "NmslibReuseIndex"]),
      ("ann_benchmarks.algorithms.falconn",
          ["FALCONN"]),
      ("ann_benchmarks.algorithms.balltree",
          ["BallTree"]),
      ("ann_benchmarks.algorithms.rpforest",
          ["RPForest"]),
      ("ann_benchmarks.algorithms.bruteforce",
          ["BruteForce", "BruteForceBLAS"]),
      ("ann_benchmarks.algorithms.subprocess",
          ["BitSubprocess", "BitSubprocessPrepared", "IntSubprocess", "FloatSubprocess"]),
      ("ann_benchmarks.algorithms.faiss",
          ["FaissLSH"]),
      ("ann_benchmarks.algorithms.dolphinnpy",
	  ["DolphinnPy"])
    ]
    constructors = {}
    for name, symbols in constructors_:
        try:
            module = importlib.import_module(name)
            for symbol in symbols:
                assert hasattr(module, symbol), """\
import error: module %s does not define symbol %s""" % (name, symbol)
                constructors[symbol] = getattr(module, symbol)
        except ImportError:
            try:
                t, v, tb = sys.exc_info()
                traceback.print_exception(t, v, tb)
            finally:
                del tb
            print """\
warning: module %s could not be loaded, some algorithm constructors will not \
be available""" % name
            for symbol in symbols:
                constructors[symbol] = None

    manifest, X = get_dataset(args.dataset, args.limit)
    if not args.query_dataset:
        X_train, X_test = split_dataset(X)
    else:
        X_train = X
        query_manifest, X_test = get_dataset(args.query_dataset)
        assert manifest == query_manifest, """\
error: the training dataset and query dataset have incompatible manifests"""

    results_fn = get_fn('results', args.dataset, args.limit)
    queries_fn = get_query_cache_path(
        args.dataset, args.limit, args.distance, args.query_dataset)

    print('storing queries in', queries_fn, 'and results in', results_fn)

    if not os.path.exists(queries_fn):
        queries = compute_distances(args.distance, X_train, X_test)
        with open(queries_fn, 'w') as f:
            pickle.dump(queries, f)
    else:
        with open(queries_fn) as f:
            queries = pickle.load(f)

    print('got', len(queries), 'queries')

    algos_already_ran = set()
    if os.path.exists(results_fn) and not args.force:
        for line in open(results_fn):
            run = json.loads(line)
            algos_already_ran.add((run["library"], run["name"]))

    point_type = manifest['point_type']
    algos = get_algorithms(
        definitions, constructors, len(X_train[0]), point_type, args.distance)

    if args.algorithm:
        print('running only', args.algorithm)
        algos = {args.algorithm: algos[args.algorithm]}
        if args.sub_algorithm:
            algos[args.algorithm] = \
              [algo for algo in algos[args.algorithm] if algo.name == args.sub_algorithm]

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
            args=(X_train, queries, library, algo, args.distance, results_fn, args.runs, args.single))
        p.start()
        p.join()

if __name__ == '__main__':
    main()
