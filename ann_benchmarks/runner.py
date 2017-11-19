import multiprocessing
import time


def run(count, X_train, X_test, library, algo, distance, result_pipe,
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
