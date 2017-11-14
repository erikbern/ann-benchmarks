from __future__ import absolute_import
import time, os, multiprocessing, argparse, resource, random
import sys
import shutil

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.results import get_results, store_results
from ann_benchmarks.distance import metrics as pd
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.definitions import get_algorithms, get_definitions
from ann_benchmarks.algorithms.constructors import available_constructors as constructors

def run_algo(count, X_train, X_test, library, algo, distance, result_pipe,
        run_count=3, force_single=False, use_batch_query=False):
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
        for i in range(run_count):
            def single_query(v):
                if prepared_queries:
                    algo.prepare_query(v, count)
                    start = time.time()
                    algo.run_prepared_query()
                    total = (time.time() - start)
                    candidates = algo.get_prepared_query_results()
                else:
                    start = time.time()
                    candidates = algo.query(v, count)
                    total = (time.time() - start)
                candidates = [(int(idx), float(pd[distance]['distance'](v, X_train[idx])))
                              for idx in candidates]
                if len(candidates) > count:
                    print('warning: algorithm %s returned %d results, but count is only %d)' % (algo.name, len(candidates), count))
                return (total, candidates)

            def batch_query(X):
                start = time.time()
                result = algo.batch_query(X, count)
                total = (time.time() - start)
                candidates = [map(
                    lambda idx: (int(idx), float(pd[distance]['distance'](X[i], X_train[idx]))),
                    result[i]) for i in range(len(X))]
                return [(total / float(len(X)), v) for v in candidates]

            if use_batch_query:
                results = batch_query(X_test)
            elif algo.use_threads() and not force_single:
                pool = multiprocessing.pool.ThreadPool()
                results = pool.map(single_query, X_test)
            else:
                results = map(single_query, X_test)

            total_time = sum(time for time, _ in results)
            total_candidates = sum(len(candidates) for _, candidates in results)
            search_time = total_time / len(X_test)
            avg_candidates = total_candidates / len(X_test)
            best_search_time = min(best_search_time, search_time)

        verbose = hasattr(algo, "query_verbose")
        attrs = {
            "library": library,
            "name": algo.name,
            "build_time": build_time,
            "best_search_time": best_search_time,
            "index_size": index_size,
            "candidates": avg_candidates,
            "run_count": run_count,
            "run_alone": force_single,
            "expect_extra": verbose,
            "batch_mode": use_batch_query
        }
        result_pipe.send((attrs, results))
        if verbose:
            metadata = \
                [m for _, m in [algo.query_verbose(q, count) for q, _, _ in queries]]
            result_pipe.send(metadata)
    finally:
        algo.done()


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
            "-k", "--count",
            default=10,
            type=positive_int,
            help="the number of near neighbours to search for")
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
            help='''re-run algorithms even if their results already exist''',
            action='store_true')
    parser.add_argument(
            '--runs',
            metavar='COUNT',
            type=positive_int,
            help='run each algorithm instance %(metavar)s times and use only the best result',
            default=3)
    parser.add_argument(
            '--timeout',
            type=int,
            help='Timeout (in seconds) for each individual algorithm run, or -1 if no timeout should be set',
            default=-1)
    parser.add_argument(
            '--single',
            help='run only a single algorithm instance at a time',
            action='store_true')
    parser.add_argument(
            '--batch',
            help='Provide Queryset as Batch',
            action='store_true')
    parser.add_argument(
            '--no_save_index',
            help='do not save indices',
            action='store_true')

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None

    definitions = get_definitions(args.definitions)
    if hasattr(args, "list_algorithms"):
        print('The following algorithms are supported...')
        for point in definitions:
            print('\t... for the point type "%s"...' % point)
            for metric in definitions[point]:
                print('\t\t... and the distance metric "%s":' % metric)
                for algorithm in definitions[point][metric]:
                    print('\t\t\t%s' % algorithm)
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

    dataset = get_dataset(args.dataset)
    X_train = dataset['train']
    X_test = dataset['test']
    print('got a train set of size (%d * %d)' % X_train.shape)
    print('got %d queries' % len(X_test))

    algos_already_run = set()
    if not args.force:
        for run in get_results(args.dataset, args.limit, args.count,
                args.distance):
            algos_already_run.add((run.attrs["library"], run.attrs["name"]))

    point_type = 'float' # TODO(erikbern): should look at the type of X_train
    algos = get_algorithms(definitions, constructors,
        len(X_train[0]), point_type, args.distance, args.count)

    if args.algorithm:
        print('running only', args.algorithm)
        algos = {args.algorithm: algos[args.algorithm]}
        if args.sub_algorithm:
            algos[args.algorithm] = \
              [algo for algo in algos[args.algorithm] if algo.name == args.sub_algorithm]

    algos_flat = []

    for library in algos.keys():
        for algo in algos[library]:
            if (library, algo.name) not in algos_already_run:
                algos_flat.append((library, algo))

    random.shuffle(algos_flat)

    print('order:', [a.name for l, a in algos_flat])

    for library, algo in algos_flat:
        recv_pipe, send_pipe = multiprocessing.Pipe(duplex=False)
        print(algo.name, '...')
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(
            target=run_algo,
            args=(args.count, X_train, X_test, library, algo,
                  args.distance, send_pipe, args.runs, args.single, args.batch))

        p.start()
        send_pipe.close()

        timed_out = False
        try:
            r = recv_pipe.poll(args.timeout)
            if r:
                # If there's something waiting in the pipe at this point, then
                # the worker has begun sending us results and we should receive
                # them
                attrs, results = recv_pipe.recv()
                if "expect_extra" in attrs:
                    if attrs["expect_extra"]:
                        attrs["extra"] = recv_pipe.recv()
                    del attrs["expect_extra"]
            else:
                # If we've exceeded the timeout and there are no results, then
                # terminate the worker process (XXX: what should we do about
                # algo.done() here?)
                p.terminate()
                timed_out = True
                results = None
        except EOFError:
            # The worker has crashed or otherwise failed to send us results
            results = None
        p.join()
        recv_pipe.close()

        if results:
            store_results(attrs, results, args.dataset, args.limit,
                          args.count, args.distance)
        elif timed_out:
            print('algorithm worker process took too long')
        else:
            print('algorithm worker process stopped unexpectedly')
