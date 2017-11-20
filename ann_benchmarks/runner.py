import multiprocessing
import time

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.distance import metrics
from ann_benchmarks.results import get_results, store_results


def run(dataset, count, library, algo, run_count=3, force_single=False, use_batch_query=False):
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
        if algo != 'bf':
            index_size_before = algo.get_index_size("self")
            print('X_train:', X_train)
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
        store_results(attrs, results, dataset, count, distance)
    finally:
        algo.done()
