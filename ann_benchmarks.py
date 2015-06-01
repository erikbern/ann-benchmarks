import sklearn.neighbors
import annoy
import pyflann
import panns
import nearpy, nearpy.hashes, nearpy.distances
import gzip, numpy, time, os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import sklearn.cross_validation, sklearn.preprocessing

n_iter = 50
n_neighbors = 100


class BaseANN(object):
    pass

        
class LSHF(BaseANN):
    def __init__(self, n_estimators=10, n_candidates=50):
        self._lshf = sklearn.neighbors.LSHForest(n_candidates=50, n_neighbors=n_neighbors)
        self.name = 'LSHF(n_est=%d, n_cand=%d)' % (n_estimators, n_candidates)

    def fit(self, X):
        self._lshf.fit(X)

    def query(self, v, n):
        return self._lshf.kneighbors(v, return_distance=False)[0]


class FLANN(BaseANN):
    def __init__(self, target_precision):
        self._flann = pyflann.FLANN(target_precision=target_precision, algorithm='autotuned', log_level='info')
        self.name = 'FLANN(target_precision=%f)' % target_precision

    def fit(self, X):
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._flann.build_index(X)

    def query(self, v, n):
        v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        return self._flann.nn_index(v, n)[0][0]


class Annoy(BaseANN):
    def __init__(self, n_trees, n_candidates):
        self._n_trees = n_trees
        self._n_candidates = n_candidates
        self.name = 'Annoy(n_trees=%d, n_cand=%d)' % (n_trees, n_candidates)

    def fit(self, X):
        self._annoy = annoy.AnnoyIndex(f=X.shape[1], metric='angular')
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), self._n_candidates)[:n]


class PANNS(BaseANN):
    def __init__(self, n_trees, n_candidates):
        self._n_trees = n_trees
        self._n_candidates = n_candidates
        self.name = 'PANNS(n_trees=%d, n_cand=%d)' % (n_trees, n_candidates)        

    def fit(self, X):
        self._panns = panns.PannsIndex(X.shape[1], metric='angular')
        for x in X:
            self._panns.add_vector(x)
        self._panns.build(self._n_trees)

    def query(self, v, n):
        return [x for x, y in self._panns.query(v, n)]


class NearPy(BaseANN):
    def __init__(self, n_bits):
        self._n_bits = n_bits
        self.name = 'NearPy(n_bits=%d)' % (n_bits,)

    def fit(self, X):
        nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp', self._n_bits)
        self._nearpy_engine = nearpy.Engine(X.shape[1], lshashes=[nearpy_rbp], distance=nearpy.distances.CosineDistance())

        for i, x in enumerate(X):
            self._nearpy_engine.store_vector(x.tolist(), i)

    def query(self, v, n):
        return [y for x, y, z in self._nearpy_engine.neighbours(v)]


class BruteForce(BaseANN):
    def __init__(self):
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='cosine')
        self.name = 'BruteForce()'

    def fit(self, X):
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors(v, return_distance=False, n_neighbors=n)[0])


def get_dataset(which='glove'):
    local_fn = os.path.join('install', which + '.txt')
    f = open(local_fn)

    X = []
    for i, line in enumerate(f):
        v = [float(x) for x in line.strip().split()]
        X.append(v)
        if len(X) == 100000: # just for debugging purposes right now
            break

    X = numpy.vstack(X)
    X_train, X_test = sklearn.cross_validation.train_test_split(X, test_size=0.01, random_state=42)
    print X_train.shape, X_test.shape
    return X_train, X_test


bf = BruteForce()

algos = {
    'lshf': [LSHF(10, 50)],
    'flann': [FLANN(0.7), FLANN(0.8), FLANN(0.9), FLANN(0.95), FLANN(0.97), FLANN(0.98), FLANN(0.99), FLANN(0.995)],
    'panns': [PANNS(10, 100)],
    'annoy': [Annoy(3, 10), Annoy(5, 25), Annoy(10, 10), Annoy(10, 100), Annoy(20, 100), Annoy(40, 100)],
    'nearpy': [NearPy(10), NearPy(12), NearPy(15), NearPy(20)],
    'bruteforce': [bf],
}

X_train, X_test = get_dataset(which='glove')

# Prepare queries
bf.fit(X_train)
queries = []
for x in X_test:
    correct = bf.query(x, 10)
    queries.append((x, correct))

f = open('data.tsv', 'w')
for library in algos.keys():
    for algo in algos[library]:
        print library, algo, '...'
        t0 = time.time()
        if algo != 'bf':
            algo.fit(X_train)
        build_time = time.time() - t0

        for i in xrange(3): # Do multiple times to warm up page cache
            t0 = time.time()
            k = 0.0
            for v, correct in queries:
                found = algo.query(v, 10)
                k += len(set(found).intersection(correct))
            search_time = time.time() - t0
            precision = k / (len(queries) * 10)

            output = [library, algo.name, build_time, search_time, precision]
            print output
        f.write('\t'.join(map(str, output)) + '\n')
f.close()
