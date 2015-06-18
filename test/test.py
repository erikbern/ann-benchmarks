import inspect
import ann_benchmarks
from sklearn.datasets.samples_generator import make_blobs

# Generate dataset
X, labels_true = make_blobs(n_samples=1000, n_features=10,
                            centers=10, cluster_std=5,
                            random_state=0)

def check_algo(algo_name, algo):
    algo.fit(X)
    result = algo.query(X[42], 10)
    assert result[0] == 42
    assert len(result) == 10
    assert len(set(result)) == 10

def test_all_algos():
    for metric in ['angular', 'euclidean']:
        algos = ann_benchmarks.get_algos(metric)
        for algo_key in algos.keys():
            for algo in algos[algo_key]:
                yield check_algo, algo.name, algo # pass name just so unittest can capture it
