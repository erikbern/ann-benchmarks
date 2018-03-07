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

from ann_benchmarks.datasets import get_dataset, DATASETS
from ann_benchmarks.algorithms.definitions import Definition, instantiate_algorithm
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import store_results


def run_individual_query(algo, X_train, X_test, distance, count, run_count=3, force_single=False, use_batch_query=False):
    best_search_time = float('inf')
    for i in range(run_count):
        print('Run %d/%d...' % (i+1, run_count))
        n_items_processed = [0]  # a bit dumb but can't be a scalar since of Python's scoping rules

        def single_query(v):
            start = time.time()
            candidates = algo.query(v, count)
            total = (time.time() - start)
            candidates = [(int(idx), float(metrics[distance]['distance'](v, X_train[idx])))
                          for idx in candidates]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print('Processed %d/%d queries...' % (n_items_processed[0], X_test.shape[0]))
            if len(candidates) > count:
                print('warning: algorithm %s returned %d results, but count is only %d)' % (algo.name, len(candidates), count))
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
        else:
            results = [single_query(x) for x in X_test]

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
        "name": algo.name,
        "run_count": run_count,
        "run_alone": force_single,
    }
    return (attrs, results)


def run(definition, dataset, count, run_count=3, force_single=False, use_batch_query=False):
    algo = instantiate_algorithm(definition)

    D = get_dataset(dataset)
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])
    distance = D.attrs['distance']
    print('got a train set of size (%d * %d)' % X_train.shape)
    print('got %d queries' % len(X_test))

    try:
        t0 = time.time()
        index_size_before = algo.get_index_size("self")
        algo.fit(X_train)
        build_time = time.time() - t0
        index_size = algo.get_index_size("self") - index_size_before
        print('Built index in', build_time)
        print('Index size: ', index_size)

        descriptor, results = run_individual_query(algo, X_train, X_test,
                distance, count, run_count, force_single, use_batch_query)
        descriptor["build_time"] = build_time
        descriptor["index_size"] = index_size
        store_results(dataset, count, definition, descriptor, results)
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
        '--json-args',
        action='store_true')
    parser.add_argument(
        '-a', '--arg',
        dest='args', action='append')
    args = parser.parse_args()
    if args.json_args:
        algo_args = [json.loads(arg) for arg in args.args]
    else:
        algo_args = args.args

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None, # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args
    )
    run(definition, args.dataset, args.count)


def run_docker(definition, dataset, count, runs, timeout=3*3600, mem_limit=None):
    cmd = ['--dataset', dataset,
           '--algorithm', definition.algorithm,
           '--module', definition.module,
           '--constructor', definition.constructor,
           '--count', str(count),
           '--json-args']
    for arg in definition.arguments:
        cmd += ['--arg', json.dumps(arg)]
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
