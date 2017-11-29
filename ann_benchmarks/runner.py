import datetime
import docker
import multiprocessing.pool
import os
import requests
import sys
import time

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.algorithms.definitions import instantiate_algorithm
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import get_results, store_results


def run(definition, dataset, count, run_count=3, force_single=False, use_batch_query=False):
    algo = instantiate_algorithm(definition)

    D = get_dataset(dataset)
    X_train = D['train']
    X_test = D['test']
    distance = D.attrs['distance']
    print('got a train set of size (%d * %d)' % X_train.shape)
    print('got %d queries' % len(X_test))

    try:
        prepared_queries = False
        if hasattr(algo, "supports_prepared_queries"):
            prepared_queries = algo.supports_prepared_queries()

        t0 = time.time()
        print('X_train:', X_train)
        algo.fit(X_train)
        build_time = time.time() - t0
        print('Built index in', build_time)

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
                candidates = [(int(idx), float(metrics[distance]['distance'](v, X_train[idx])))
                              for idx in candidates]
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
            "name": algo.name,
            "build_time": build_time,
            "best_search_time": best_search_time,
            "candidates": avg_candidates,
            "run_count": run_count,
            "run_alone": force_single,
            "expect_extra": verbose,
            "batch_mode": use_batch_query
        }
        store_results(attrs, results, dataset, count, distance)
    finally:
        algo.done()


def run_docker(definition, dataset, count, runs, timeout=7200):
    cmd = '--dataset %s --module %s --constructor %s --count %d %s' % (
        dataset, definition.module, definition.constructor, count, # TODO: include runs
        ' '.join('--arg %s' % arg for arg in definition.arguments)
    )
    client = docker.from_env()
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath('ann_benchmarks'): {'bind': '/home/app/ann_benchmarks', 'mode': 'ro'},
            os.path.abspath('data'): {'bind': '/home/app/data', 'mode': 'ro'},
            os.path.abspath('results'): {'bind': '/home/app/results', 'mode': 'rw'},
        },
        detach=True)
    t = t0 = datetime.datetime.now()
    while True:
        exit_code = None
        try:
            exit_code = container.wait(timeout=10)
        except requests.exceptions.ConnectionError:
            pass

        # Print any logs since last timestamp
        logs = container.logs(since=t)
        sys.stdout.buffer.write(logs)
        sys.stdout.buffer.flush()
        t = datetime.datetime.now()

        # Exit if exit code
        if exit_code == 0:
            return
        elif exit_code is not None:
            raise Exception('Child process raised exception %d' % exit_code)

        # Break if we've spent too much time
        if (t - t0).total_seconds() > timeout:
            raise Exception('Child process time limit %fs exceeded' % timeout)
