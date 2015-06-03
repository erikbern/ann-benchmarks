import sklearn.neighbors
import annoy
import pyflann
import panns
import nearpy, nearpy.hashes, nearpy.distances
import pykgraph
import gzip, numpy, time, os, multiprocessing
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import sklearn.cross_validation, sklearn.preprocessing, random

class BaseANN(object):
    pass

        
class LSHF(BaseANN):
    def __init__(self, n_estimators=10, n_candidates=50):
        self.name = 'LSHF(n_est=%d, n_cand=%d)' % (n_estimators, n_candidates)
        self._n_estimators = n_estimators
        self._n_candidates = n_candidates

    def fit(self, X):
        self._lshf = sklearn.neighbors.LSHForest(n_estimators=self._n_estimators, n_candidates=self._n_candidates)
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._lshf.fit(X)

    def query(self, v, n):
        v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        return self._lshf.kneighbors(v, return_distance=False, n_neighbors=n)[0]


class BallTree(BaseANN):
    def __init__(self, leaf_size=20):
        self.name = 'BallTree(leaf_size=%d)' % leaf_size
        self._leaf_size = leaf_size

    def fit(self, X):
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._tree = sklearn.neighbors.BallTree(X, leaf_size=self._leaf_size)

    def query(self, v, n):
        v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        dist, ind = self._tree.query(v, k=n)
        return ind[0]


class KDTree(BaseANN):
    def __init__(self, leaf_size=20):
        self.name = 'KDTree(leaf_size=%d)' % leaf_size
        self._leaf_size = leaf_size

    def fit(self, X):
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._tree = sklearn.neighbors.KDTree(X, leaf_size=self._leaf_size)

    def query(self, v, n):
        v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        dist, ind = self._tree.query(v, k=n)
        return ind[0]


class FLANN(BaseANN):
    def __init__(self, target_precision):
        self._target_precision = target_precision
        self.name = 'FLANN(target_precision=%f)' % target_precision

    def fit(self, X):
        self._flann = pyflann.FLANN(target_precision=self._target_precision, algorithm='autotuned', log_level='info')
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


class KGraph(BaseANN):
    def __init__(self, P):
        self.name = 'KGraph(P=%d)' % P
        self._P = P

    def fit(self, X):
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._kgraph = pykgraph.KGraph()
        self._kgraph.build(X, iterations=30, L=100, delta=0.002, recall=0.99, K=25)
        self._X = X # ???

    def query(self, v, n):
        v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        result = self._kgraph.search(self._X, numpy.array([v]), K=n, threads=1, P=self._P)
        return result[0]


class BruteForce(BaseANN):
    def __init__(self):
        self.name = 'BruteForce()'

    def fit(self, X):
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='cosine')
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
        #if len(X) == 100: # just for debugging purposes right now
        #    break

    X = numpy.vstack(X)
    X_train, X_test = sklearn.cross_validation.train_test_split(X, test_size=1000, random_state=42)
    print X_train.shape, X_test.shape
    return X_train, X_test

def run_algo(library, algo):
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
        search_time = (time.time() - t0) / len(queries)
        precision = k / (len(queries) * 10)

        output = [library, algo.name, build_time, search_time, precision]
        print output

    f = open('data.tsv', 'a')
    f.write('\t'.join(map(str, output)) + '\n')
    f.close()

bf = BruteForce()

algos = {
    'lshf': [LSHF(5, 10), LSHF(5, 20), LSHF(10, 20), LSHF(10, 50), LSHF(20, 100)],
    'flann': [FLANN(0.2), FLANN(0.5), FLANN(0.7), FLANN(0.8), FLANN(0.9), FLANN(0.95), FLANN(0.97), FLANN(0.98), FLANN(0.99), FLANN(0.995)],
    'panns': [PANNS(5, 20), PANNS(10, 10), PANNS(10, 50), PANNS(10, 100), PANNS(20, 100), PANNS(40, 100)],
    'annoy': [Annoy(3, 10), Annoy(5, 25), Annoy(10, 10), Annoy(10, 40), Annoy(10, 100), Annoy(10, 200), Annoy(10, 400), Annoy(10, 1000), Annoy(20, 20), Annoy(20, 100), Annoy(20, 200), Annoy(20, 400), Annoy(40, 40), Annoy(40, 100), Annoy(40, 400), Annoy(100, 100), Annoy(100, 200), Annoy(100, 400), Annoy(100, 1000)],
    'nearpy': [NearPy(10), NearPy(12), NearPy(15), NearPy(20)],
    'kgraph': [KGraph(20), KGraph(50), KGraph(100), KGraph(200), KGraph(500), KGraph(1000)],
    'bruteforce': [bf],
    'ball': [BallTree(10), BallTree(20), BallTree(40), BallTree(100), BallTree(200), BallTree(400), BallTree(1000)],
    'kd': [KDTree(10), KDTree(20), KDTree(40), KDTree(100), KDTree(200), KDTree(400), KDTree(1000)]
}

X_train, X_test = get_dataset(which='glove')

# Prepare queries
bf.fit(X_train)
queries = []
for x in X_test:
    correct = bf.query(x, 10)
    queries.append((x, correct))
    if len(queries) % 100 == 0:
        print len(queries), '...'

algos_already_ran = set()
if os.path.exists('data.tsv'):
    for line in open('data.tsv'):
        algos_already_ran.add(line.strip().split('\t')[1])

algos_flat = []

for library in algos.keys():
    for algo in algos[library]:
        if algo.name not in algos_already_ran:
            algos_flat.append((library, algo))

random.shuffle(algos_flat)

for library, algo in algos_flat:
    print algo.name, '...'
    # Spawn a subprocess to force the memory to be reclaimed at the end
    p = multiprocessing.Process(target=run_algo, args=(library, algo))
    p.start()
    p.join()
