import random
import inspect
import ann_benchmarks
from sklearn.datasets.samples_generator import make_blobs

# Generate dataset
X, labels_true = make_blobs(n_samples=10000, n_features=10,
                            centers=10, cluster_std=5,
                            random_state=0)

def check_algo(algo_name, algo):
    algo.fit(X)
    result = algo.query(X[42], 10)
    if len(result) != 10:
        raise AssertionError('Expected results to have length 10: Result: %s' % result)
    if len(set(result)) != 10:
        raise AssertionError('Expected results to be unique: Result: %s' % result)
    #if result[0] != 42:
    #    raise AssertionError('Expected first item to be 42: Result: %s' % result)


def test_all_algos():
    for metric in ['angular', 'euclidean']:
        algos = ann_benchmarks.get_algos(metric)
        for algo_key in algos.keys():
            algo = random.choice(algos[algo_key]) # Just pick one of each
            yield check_algo, algo.name, algo # pass name just so unittest can capture it
