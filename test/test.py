import random
import inspect
import ann_benchmarks
from sklearn.datasets.samples_generator import make_blobs
import warnings
import multiprocessing

# Generate dataset
X, labels_true = make_blobs(n_samples=10000, n_features=12,
                            centers=10, cluster_std=5,
                            random_state=0)

def fit_and_query(algo, result_queue):
    # Run algorithm in subprocess in case it segfaults
    algo.fit(X)
    result = algo.query(X[42], 10)
    result_queue.put(result)


def check_algo(algo_name, algo):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=fit_and_query, args=(algo, queue))
    p.start()
    p.join()
    result = queue.get(timeout=10.0)

    if len(result) != 10:
        warnings.warn('%s: Expected results to have length 10: Result: %s' % (algo_name, result))
    if len(set(result)) != 10:
        warnings.warn('%s: Expected results to be unique: Result: %s' % (algo_name, result))
    if result[0] != 42:
        warnings.warn('%s: Expected first item to be 42: Result: %s' % (algo_name, result))


def test_all_algos():
    for metric in ['angular', 'euclidean']:
        algos = ann_benchmarks.get_algos(metric,False) # false means: don't save any indices
        for algo_key in algos.keys():
            algo = random.choice(algos[algo_key]) # Just pick one of each
            yield check_algo, algo.name, algo # pass name just so unittest can capture it
