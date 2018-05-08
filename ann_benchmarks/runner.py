from __future__ import print_function
__true_print = print

import argparse
import datetime
import docker
import json
import multiprocessing.pool
import numpy
import os
import psutil
import requests
import sys
import threading
import time
import psutil

def print(*args, **kwargs):
    __true_print(*args, **kwargs)
    sys.stdout.flush()

from ann_benchmarks.datasets import get_dataset, DATASETS
from ann_benchmarks.algorithms.definitions import Definition, instantiate_algorithm
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import store_results


def run_individual_query(algo, X_train, X_test, distance, count, run_count, force_single, use_batch_query):
    prepared_queries = False
    if hasattr(algo, "supports_prepared_queries"):
        prepared_queries = algo.supports_prepared_queries()

    best_search_time = float('inf')
    for i in range(run_count):
        print('Run %d/%d...' % (i+1, run_count))
        n_items_processed = [0]  # a bit dumb but can't be a scalar since of Python's scoping rules

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
            candidates = [(int(idx), float(metrics[distance]['distance'](v, X_train[idx])))
                          for idx in candidates]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print('Processed %d/%d queries...' % (n_items_processed[0], X_test.shape[0]))
            if len(candidates) > count:
                print('warning: algorithm %s returned %d results, but count is only %d)' % (algo, len(candidates), count))
            return (total, candidates)

        def batch_query(X):
            start = time.time()
            result = algo.batch_query(X, count)
            total = (time.time() - start)
            candidates = [[(int(idx), float(metrics[distance]['distance'](v, X_train[idx])))
                           for idx in single_results]
                          for v, single_results in zip(X, results)]
            return [(total / float(len(X)), v) for v in candidates]

        if use_batch_query:
            results = batch_query(X_test)
        elif algo.use_threads() and not force_single:
            pool = multiprocessing.pool.ThreadPool()
            results = pool.map(single_query, X_test)
            pool.close()
        else:
            p = psutil.Process()
            initial_affinity = p.cpu_affinity()
            p.cpu_affinity([initial_affinity[len(initial_affinity) // 2]]) # one of the available virtual CPU cores

            results = [single_query(x) for x in X_test]

            p.cpu_affinity(initial_affinity)

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": use_batch_query,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "run_alone": force_single,
        "distance": distance,
        "count": int(count)
    }
    return (attrs, results)


def run(definition, dataset, count, run_count=3, force_single=True, use_batch_query=False):
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups \
            or hasattr(algo, "set_query_arguments"), """\
error: query argument groups have been specified for %s.%s(%s), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function""" % (definition.module, definition.constructor, definition.arguments)

    D = get_dataset(dataset)
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])
    distance = D.attrs['distance']
    print('got a train set of size (%d * %d)' % X_train.shape)
    print('got %d queries' % len(X_test))

    try:
        prepared_queries = False
        if hasattr(algo, "supports_prepared_queries"):
            prepared_queries = algo.supports_prepared_queries()

        t0 = time.time()
        memory_usage_before = algo.get_memory_usage()
        algo.fit(X_train)
        build_time = time.time() - t0
        index_size = algo.get_memory_usage() - memory_usage_before
        print('Built index in', build_time)
        print('Index size: ', index_size)

        query_argument_groups = definition.query_argument_groups
        # Make sure that algorithms with no query argument groups still get run
        # once by providing them with a single, empty, harmless group
        if not query_argument_groups:
            query_argument_groups = [[]]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." %
                    (pos, len(query_argument_groups)))
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            descriptor, results = run_individual_query(algo, X_train, X_test,
                    distance, count, run_count, force_single, use_batch_query)
            descriptor["build_time"] = build_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = definition.algorithm
            descriptor["dataset"] = dataset
            store_results(dataset,
                    count, definition, query_arguments, descriptor, results)
    finally:
        algo.done()


def run_from_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--algorithm',
        required=True)
    parser.add_argument(
        '--module',
        required=True)
    parser.add_argument(
        '--constructor',
        required=True)
    parser.add_argument(
        '--count',
        required=True,
        type=int)
    parser.add_argument(
        'build')
    parser.add_argument(
        'queries',
        nargs='*',
        default=[])
    args = parser.parse_args()
    algo_args = json.loads(args.build)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None, # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args
    )
    run(definition, args.dataset, args.count)


def run_docker(definition, dataset, count, runs, timeout=3*3600, mem_limit=None):
    cmd = ['--dataset', dataset,
           '--algorithm', definition.algorithm,
           '--module', definition.module,
           '--constructor', definition.constructor,
           '--count', str(count)]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]
    print('Running command', cmd)
    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    print('Memory limit:', mem_limit)
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath('ann_benchmarks'): {'bind': '/home/app/ann_benchmarks', 'mode': 'ro'},
            os.path.abspath('data'): {'bind': '/home/app/data', 'mode': 'ro'},
            os.path.abspath('results'): {'bind': '/home/app/results', 'mode': 'rw'},
        },
        mem_limit=mem_limit,
        detach=True)

    def stream_logs():
        import colors
        for line in container.logs(stream=True):
            print(colors.color(line.decode().rstrip(), fg='yellow'))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()
    try:
        exit_code = container.wait(timeout=timeout)

        # Exit if exit code
        if exit_code == 0:
            return
        elif exit_code is not None:
            raise Exception('Child process raised exception %d' % exit_code)

    finally:
        container.remove(force=True)
