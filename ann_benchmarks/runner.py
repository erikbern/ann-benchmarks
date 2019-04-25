from __future__ import print_function
__true_print = print  # noqa

import argparse
import datetime
import docker
import json
import multiprocessing
import numpy
import os
import psutil
import requests
import sys
import threading
import time


def print(*args, **kwargs):  # noqa
    __true_print(*args, **kwargs)
    sys.stdout.flush()


from ann_benchmarks.datasets import get_dataset, DATASETS
from ann_benchmarks.algorithms.definitions import (Definition,
                                                   instantiate_algorithm,
                                                   get_algorithm_name)
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import store_results


def run_individual_query(algo, X_train, X_test, distance, count, run_count,
                         batch):
    prepared_queries = \
        (batch and hasattr(algo, "prepare_batch_query")) or \
        ((not batch) and hasattr(algo, "prepare_query"))

    best_search_time = float('inf')
    for i in range(run_count):
        print('Run %d/%d...' % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

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
            candidates = [(int(idx), float(metrics[distance]['distance'](v, X_train[idx])))  # noqa
                          for idx in candidates]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print('Processed %d/%d queries...' %
                      (n_items_processed[0], X_test.shape[0]))
            if len(candidates) > count:
                print('warning: algorithm %s returned %d results, but count'
                      ' is only %d)' % (algo, len(candidates), count))
            return (total, candidates)

        def batch_query(X):
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.time()
                algo.run_batch_query()
                total = (time.time() - start)
            else:
                start = time.time()
                algo.batch_query(X, count)
                total = (time.time() - start)
            results = algo.get_batch_results()
            candidates = [[(int(idx), float(metrics[distance]['distance'](v, X_train[idx])))  # noqa
                           for idx in single_results]
                          for v, single_results in zip(X, results)]
            return [(total / float(len(X)), v) for v in candidates]

        if batch:
            results = batch_query(X_test)
        else:
            results = [single_query(x) for x in X_test]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count)
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def run(definition, dataset, count, run_count, batch):
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
            descriptor, results = run_individual_query(
                algo, X_train, X_test, distance, count, run_count, batch)
            descriptor["build_time"] = build_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = get_algorithm_name(
                definition.algorithm, batch)
            descriptor["dataset"] = dataset
            store_results(dataset, count, definition,
                          query_arguments, descriptor, results, batch)
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
        '--runs',
        required=True,
        type=int)
    parser.add_argument(
        '--batch',
        action='store_true')
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
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)


def run_docker(definition, dataset, count, runs, timeout, batch,
               mem_limit=None):
    import colors  # Think it doesn't work in Python 2

    cmd = ['--dataset', dataset,
           '--algorithm', definition.algorithm,
           '--module', definition.module,
           '--constructor', definition.constructor,
           '--runs', str(runs),
           '--count', str(count)]
    if batch:
        cmd += ['--batch']
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]
    print('Running command', cmd)
    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    print('Memory limit:', mem_limit)
    cpu_limit = "0-%d" % (multiprocessing.cpu_count() - 1)
    if not batch:
        # Limit to first cpu if not in batch mode
        cpu_limit = "0"
    print('Running on CPUs:', cpu_limit)

    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath('ann_benchmarks'):
                {'bind': '/home/app/ann_benchmarks', 'mode': 'ro'},
            os.path.abspath('data'):
                {'bind': '/home/app/data', 'mode': 'ro'},
            os.path.abspath('results'):
                {'bind': '/home/app/results', 'mode': 'rw'},
        },
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True)

    def stream_logs():
        for line in container.logs(stream=True):
            print(colors.color(line.decode().rstrip(), fg='blue'))

    if sys.version_info >= (3, 0):
        t = threading.Thread(target=stream_logs, daemon=True)
    else:
        t = threading.Thread(target=stream_logs)
        t.daemon = True
    t.start()
    try:
        exit_code = container.wait(timeout=timeout)

        # Exit if exit code
        if exit_code == 0:
            return
        elif exit_code is not None:
            print(colors.color(container.logs().decode(), fg='red'))
            raise Exception('Child process raised exception %d' % exit_code)

    finally:
        container.remove(force=True)
