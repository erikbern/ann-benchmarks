import gzip, numpy, time, os, multiprocessing, argparse, pickle, resource, random, math
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import sklearn.preprocessing

# Set resource limits to prevent memory bombs
memory_limit = 12 * 2**30
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
if soft == resource.RLIM_INFINITY or soft >= memory_limit:
    print('resetting memory limit from', soft, 'to', memory_limit)
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))

os.environ['OMP_THREAD_LIMIT'] = '1' # just to limit number of processors                                                                                                                                                                    

# Nmslib specific code
# Remove old indices stored on disk
INDEX_DIR='indices'    
#import shutil
#if os.path.exists(INDEX_DIR):
#  shutil.rmtree(INDEX_DIR)

class BaseANN(object):
    pass

        
class FALCONN(BaseANN):
    def __init__(self, metric, num_bits, num_tables, num_probes):
        self.name = 'FALCONN(K={}, L={}, T={})'.format(num_bits, num_tables, num_probes)
        self._metric = metric
        self._num_bits = num_bits
        self._num_tables = num_tables
        self._num_probes = num_probes
        self._center = None
        self._params = None
        self._index = None
        self._buf = None

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if self._metric == 'angular':
            X /= numpy.linalg.norm(X, axis=1).reshape(-1,  1)
        self._center = numpy.mean(X, axis=0)
        X -= self._center
        import falconn
        self._params = falconn.LSHConstructionParameters()
        self._params.dimension = X.shape[1]
        self._params.distance_function = 'euclidean_squared'
        self._params.lsh_family = 'cross_polytope'
        falconn.compute_number_of_hash_functions(self._num_bits, self._params)
        self._params.l = self._num_tables
        self._params.num_rotations = 1
        self._params.num_setup_threads = 0
        self._params.storage_hash_table = 'flat_hash_table'
        self._params.seed = 95225714
        self._index = falconn.LSHIndex(self._params)
        self._index.setup(X)
        self._index.set_num_probes(self._num_probes)
        self._buf = numpy.zeros((X.shape[1],), dtype=numpy.float32)

    def query(self, v, n):
        numpy.copyto(self._buf, v)
        if self._metric == 'angular':
            self._buf /= numpy.linalg.norm(self._buf)
        self._buf -= self._center
        return self._index.find_k_nearest_neighbors(self._buf, n)

class LSHF(BaseANN):
    def __init__(self, metric, n_estimators=10, n_candidates=50):
        self.name = 'LSHF(n_est=%d, n_cand=%d)' % (n_estimators, n_candidates)
        self._metric = metric
        self._n_estimators = n_estimators
        self._n_candidates = n_candidates

    def fit(self, X):
        import sklearn.neighbors
        self._lshf = sklearn.neighbors.LSHForest(n_estimators=self._n_estimators, n_candidates=self._n_candidates)
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._lshf.fit(X)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        return self._lshf.kneighbors(v, return_distance=False, n_neighbors=n)[0]


class BallTree(BaseANN):
    def __init__(self, metric, leaf_size=20):
        self.name = 'BallTree(leaf_size=%d)' % leaf_size
        self._leaf_size = leaf_size
        self._metric = metric

    def fit(self, X):
        import sklearn.neighbors
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._tree = sklearn.neighbors.BallTree(X, leaf_size=self._leaf_size)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        dist, ind = self._tree.query(v, k=n)
        return ind[0]


class KDTree(BaseANN):
    def __init__(self, metric, leaf_size=20):
        self.name = 'KDTree(leaf_size=%d)' % leaf_size
        self._leaf_size = leaf_size
        self._metric = metric

    def fit(self, X):
        import sklearn.neighbors
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._tree = sklearn.neighbors.KDTree(X, leaf_size=self._leaf_size)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        dist, ind = self._tree.query(v, k=n)
        return ind[0]


class FLANN(BaseANN):
    def __init__(self, metric, target_precision):
        self._target_precision = target_precision
        self.name = 'FLANN(target_precision=%f)' % target_precision
        self._metric = metric

    def fit(self, X):
        import pyflann
        self._flann = pyflann.FLANN(target_precision=self._target_precision, algorithm='autotuned', log_level='info')
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._flann.build_index(X)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        return self._flann.nn_index(v, n)[0][0]


class Annoy(BaseANN):
    def __init__(self, metric, n_trees, search_k):
        self._n_trees = n_trees
        self._search_k = search_k
        self._metric = metric
        self.name = 'Annoy(n_trees=%d, search_k=%d)' % (n_trees, search_k)

    def fit(self, X):
        import annoy
        self._annoy = annoy.AnnoyIndex(f=X.shape[1], metric=self._metric)
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)


class PANNS(BaseANN):
    def __init__(self, metric, n_trees, n_candidates):
        self._n_trees = n_trees
        self._n_candidates = n_candidates
        self._metric = metric
        self.name = 'PANNS(n_trees=%d, n_cand=%d)' % (n_trees, n_candidates)        

    def fit(self, X):
        import panns
        self._panns = panns.PannsIndex(X.shape[1], metric=self._metric)
        for x in X:
            self._panns.add_vector(x)
        self._panns.build(self._n_trees)

    def query(self, v, n):
        return [x for x, y in self._panns.query(v, n)]


class NearPy(BaseANN):
    def __init__(self, metric, n_bits, hash_counts):
        self._n_bits = n_bits
        self._hash_counts = hash_counts
        self._metric = metric
        self.name = 'NearPy(n_bits=%d, hash_counts=%d)' % (n_bits, hash_counts)

    def fit(self, X):
        import nearpy

        hashes = []

        for k in xrange(self._hash_counts):
            nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, self._n_bits)
            hashes.append(nearpy_rbp)

        if self._metric == 'euclidean':
            dist = nearpy.distances.EuclideanDistance()
            self._nearpy_engine = nearpy.Engine(X.shape[1], lshashes=hashes, distance=dist)
        else: # Default (angular) = Cosine distance
            self._nearpy_engine = nearpy.Engine(X.shape[1], lshashes=hashes)

        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        for i, x in enumerate(X):
            self._nearpy_engine.store_vector(x.tolist(), i)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize(v, axis=1, norm='l2')[0]
        return [y for x, y, z in self._nearpy_engine.neighbours(v)]


class KGraph(BaseANN):
    def __init__(self, metric, P, index_params, save_index):
        self.name = 'KGraph(%s,P=%d)' % (metric, P)
        self._P = P
        self._metric = metric
        self._index_params = index_params
        self._save_index = save_index

    def fit(self, X):
        os.environ['OMP_THREAD_LIMIT'] = '40'
        import pykgraph

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        #if self._metric == 'angular':
        #    X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._kgraph = pykgraph.KGraph(X, self._metric)
        path = os.path.join(INDEX_DIR, 'kgraph-index-%s' % self._metric)
        if os.path.exists(path):
            self._kgraph.load(path)
        else:
            self._kgraph.build(**self._index_params) #iterations=30, L=100, delta=0.002, recall=0.99, K=25)
            if not os.path.exists(INDEX_DIR):
              os.makedirs(INDEX_DIR)
            if self._save_index:
              self._kgraph.save(path)
        os.environ['OMP_THREAD_LIMIT'] = '1'

    def query(self, v, n):
        if v.dtype != numpy.float32:
            v = v.astype(numpy.float32)
        result = self._kgraph.search(numpy.array([v]), K=n, threads=1, P=self._P)
        return result[0]

class NmslibReuseIndex(BaseANN):
    def __init__(self, metric, method_name, index_param, save_index, query_param):
        self._nmslib_metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]
        self._method_name = method_name
        self._save_index = save_index
        self._index_param = index_param
        self._query_param = query_param
        self.name = 'Nmslib(method_name=%s, index_param=%s, query_param=%s)' % (method_name, index_param, query_param)
        self._index_name = os.path.join(INDEX_DIR, "nmslib_%s_%s_%s" % (self._method_name, metric, '_'.join(self._index_param))) 

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
          os.makedirs(d)

    def fit(self, X):
        os.environ['OMP_THREAD_LIMIT'] = '40'
        import nmslib_vector
        if self._method_name == 'vptree':
            # To avoid this issue:
            # terminate called after throwing an instance of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too big. Select the parameters so that <total # of records> is NOT less than <bucket size> * 1000
            # Aborted (core dumped)
            self._index_param.append('bucketSize=%d' % min(int(X.shape[0] * 0.0005), 1000))
                                        
        self._index = nmslib_vector.init(self._nmslib_metric, [], self._method_name, nmslib_vector.DataType.VECTOR, nmslib_vector.DistType.FLOAT)
    
        for i, x in enumerate(X):
            nmslib_vector.addDataPoint(self._index, i, x.tolist())


        if os.path.exists(self._index_name):
            print "Loading index from file"
            nmslib_vector.loadIndex(self._index, self._index_name)
        else:
            nmslib_vector.createIndex(self._index, self._index_param)
            if self._save_index: 
              nmslib_vector.saveIndex(self._index, self._index_name)

        nmslib_vector.setQueryTimeParams(self._index, self._query_param)

        os.environ['OMP_THREAD_LIMIT'] = '1'

    def query(self, v, n):
        import nmslib_vector
        return nmslib_vector.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        import nmslib_vector
        nmslib_vector.freeIndex(self._index)

class NmslibNewIndex(BaseANN):
    def __init__(self, metric, method_name, method_param):
        self._nmslib_metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]
        self._method_name = method_name
        self._method_param = method_param
        self.name = 'Nmslib(method_name=%s, method_param=%s)' % (method_name, method_param)

    def fit(self, X):
        import nmslib_vector
        if self._method_name == 'vptree':
            # To avoid this issue:
            # terminate called after throwing an instance of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too big. Select the parameters so that <total # of records> is NOT less than <bucket size> * 1000
            # Aborted (core dumped)
            self._method_param.append('bucketSize=%d' % min(int(X.shape[0] * 0.0005), 1000))
                                        
        self._index = nmslib_vector.init(self._nmslib_metric, [], self._method_name, nmslib_vector.DataType.VECTOR, nmslib_vector.DistType.FLOAT)
    
        for i, x in enumerate(X):
            nmslib_vector.addDataPoint(self._index, i, x.tolist())

        nmslib_vector.createIndex(self._index, self._method_param)

    def query(self, v, n):
        import nmslib_vector
        return nmslib_vector.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        import nmslib_vector
        nmslib_vector.freeIndex(self._index)


class RPForest(BaseANN):
    def __init__(self, leaf_size, n_trees):
        from rpforest import RPForest
        self.name = 'RPForest(leaf_size=%d, n_trees=%d)' % (leaf_size, n_trees)
        self._model = RPForest(leaf_size=leaf_size, no_trees=n_trees)

    def fit(self, X):
        self._model.fit(X)

    def query(self, v, n):
        return self._model.query(v, n)


class BruteForce(BaseANN):
    def __init__(self, metric):
        self._metric = metric
        self.name = 'BruteForce()'

    def fit(self, X):
        import sklearn.neighbors
        metric = {'angular': 'cosine', 'euclidean': 'l2'}[self._metric]
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric=metric)
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors(v, return_distance=False, n_neighbors=n)[0])


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, metric, precision=numpy.float32):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
        self._metric = metric
        self._precision = precision
        self.name = 'BruteForceBLAS()'

    def fit(self, X):
        """Initialize the search index."""
        lens = (X ** 2).sum(-1)  # precompute (squared) length of each vector
        if self._metric == 'angular':
            X /= numpy.sqrt(lens)[..., numpy.newaxis]  # normalize index vectors to unit length
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'euclidean':
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!

    def query(self, v, n):
        """Find indices of `n` most similar vectors from the index to query vector `v`."""
        v = numpy.ascontiguousarray(v, dtype=self._precision)  # use same precision for query as for index
        # HACK we ignore query length as that's a constant not affecting the final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!
        indices = numpy.argpartition(dists, n)[:n]  # partition-sort by distance, get `n` closest
        return sorted(indices, key=lambda index: dists[index])  # sort `n` closest into correct order

def get_dataset(which='glove', limit=-1, random_state = 2, test_size = 10000):
    cache = 'queries/%s-%d-%d-%d.npz' % (which, test_size, limit, random_state)
    if os.path.exists(cache):
        v = numpy.load(cache)
        X_train = v['train']
        X_test = v['test']
        print(X_train.shape, X_test.shape)
        return X_train, X_test
    local_fn = os.path.join('install', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    X = []
    for i, line in enumerate(f):
        v = [float(x) for x in line.strip().split()]
        X.append(v)
        if limit != -1 and len(X) == limit:
            break

    X = numpy.vstack(X)
    import sklearn.cross_validation

    # Here Erik is most welcome to use any other random_state
    # However, it is best to use a new random seed for each major re-evaluation,
    # so that we test on a trully bind data.
    X_train, X_test = sklearn.cross_validation.train_test_split(X, test_size=test_size, random_state=random_state)
    X_train = X_train.astype(numpy.float)
    X_test = X_test.astype(numpy.float)
    print(X_train.shape, X_test.shape)
    numpy.savez(cache, train=X_train, test=X_test)
    return X_train, X_test


def run_algo(args, library, algo, results_fn):
    pool = multiprocessing.Pool()
    X_train, X_test = pool.apply(get_dataset, [args.dataset, args.limit])
    pool.close()
    pool.join()

    t0 = time.time()
    if algo != 'bf':
        algo.fit(X_train)
    build_time = time.time() - t0
    print('Built index in', build_time)

    best_search_time = float('inf')
    best_precision = 0.0 # should be deterministic but paranoid
    for i in xrange(3): # Do multiple times to warm up page cache, use fastest
        t0 = time.time()
        k = 0.0
        for v, correct in queries:
            found = algo.query(v, 10)
            k += len(set(found).intersection(correct))
        search_time = (time.time() - t0) / len(queries)
        precision = k / (len(queries) * 10)
        best_search_time = min(best_search_time, search_time)
        best_precision = max(best_precision, precision)
        print(search_time, precision)

    output = [library, algo.name, build_time, best_search_time, best_precision]
    print(output)

    f = open(results_fn, 'a')
    f.write('\t'.join(map(str, output)) + '\n')
    f.close()


def get_queries(args):
    print('computing queries with correct results...')

    bf = BruteForceBLAS(args.distance)
    X_train, X_test = get_dataset(which=args.dataset, limit=args.limit)

    # Prepare queries
    bf.fit(X_train)
    queries = []
    for x in X_test:
        correct = bf.query(x, 10)
        queries.append((x, correct))
        if len(queries) % 100 == 0:
            print(len(queries), '...')

    return queries
            
def get_algos(m, save_index):
    algos = {
        'lshf': [LSHF(m, 5, 10), LSHF(m, 5, 20), LSHF(m, 10, 20), LSHF(m, 10, 50), LSHF(m, 20, 100)],
        'flann': [FLANN(m, 0.2), FLANN(m, 0.5), FLANN(m, 0.7), FLANN(m, 0.8), FLANN(m, 0.9), FLANN(m, 0.95), FLANN(m, 0.97), FLANN(m, 0.98), FLANN(m, 0.99), FLANN(m, 0.995)],
        'panns': [PANNS(m, 5, 20), PANNS(m, 10, 10), PANNS(m, 10, 50), PANNS(m, 10, 100), PANNS(m, 20, 100), PANNS(m, 40, 100)],
        'annoy': [Annoy(m, n_trees, search_k) for n_trees in [100, 200, 400] for search_k in [100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000, 100000, 200000, 400000]],
        'nearpy': [NearPy(m, 10, 5), NearPy(m, 10, 10), NearPy(m, 10, 20), NearPy(m, 10, 40), # NearPy(m, 10, 100),
                   NearPy(m, 12, 5), NearPy(m, 12, 10), NearPy(m, 12, 20), NearPy(m, 12, 40), # NearPy(m, 12, 100),
                   NearPy(m, 14, 5), NearPy(m, 14, 10), NearPy(m, 14, 20), NearPy(m, 14, 40), # NearPy(m, 14, 100),
                   NearPy(m, 16, 5), NearPy(m, 16, 10), NearPy(m, 16, 15), NearPy(m, 16, 20), NearPy(m, 16, 25), NearPy(m, 16, 30), NearPy(m, 16, 40)], #, NearPy(m, 16, 50), NearPy(m, 16, 70), NearPy(m, 16, 90), NearPy(m, 16, 120), NearPy(m, 16, 150)],
        'bruteforce': [BruteForce(m)],
        'ball': [BallTree(m, 10), BallTree(m, 20), BallTree(m, 40), BallTree(m, 100), BallTree(m, 200), BallTree(m, 400), BallTree(m, 1000)],
        'kd': [KDTree(m, 10), KDTree(m, 20), KDTree(m, 40), KDTree(m, 100), KDTree(m, 200), KDTree(m, 400), KDTree(m, 1000)],

        # START: Non-Metric Space Library (nmslib) entries
        'bruteforce0(nmslib)': [NmslibNewIndex(m, 'seq_search', ['copyMem=0'])],
        # We don't need copyMem=1 now, because the new Python wrapper already re-creates data points.
        #'bruteforce1(nmslib)': [NmslibNewIndex(m, 'seq_search', ['copyMem=1'])],

        'BallTree(nmslib)': [],

        'hnsw(nmslib)': [],

        'SW-graph(nmslib)' :[],

        'MP-lsh(lshkit)' :[]
    }

    for r in [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
      algos['BallTree(nmslib)'].append(NmslibNewIndex(m, 'vptree', ['tuneK=10', 'desiredRecall=%f' % r]))

    if m == 'euclidean':
        # kgraph 
        kgraph_preset ={'reverse':-1};
        kgraph_Ps = [1,2,3,4,5,10,20,30,40,50,60,70,80,90,100]
        algos['kgraph'] = [KGraph(m, P, kgraph_preset, save_index) for P in kgraph_Ps]

        # nmslib algorithms
        # Only works for euclidean distance
        MsPostsEfs=[
                (32, 2, [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 300, 400]),
                (20, 2, [2, 5, 10, 15, 20, 30, 40, 50, 70, 80, 120, 200, 400]),
                (12, 0, [1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 80, 120]),
                (4,  0, [1, 2, 5, 10, 20, 30, 50, 70, 90, 120]),
                (8,  0, [1, 2, 5, 10, 20, 30, 50, 70, 90, 120, 160])  ]
        for oneCase in MsPostsEfs:
            for ef in oneCase[2]:
                algos['hnsw(nmslib)'].append(NmslibReuseIndex(m, 'hnsw', 
                                                              ['M=%d' % oneCase[0], 'post=%d' % oneCase[1], 'efConstruction=400'], save_index,
                                                              ['ef=%d' % ef]))
        
        for r in [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
          algos['MP-lsh(lshkit)'].append(NmslibNewIndex(m, 'lsh_multiprobe', ['desiredRecall=%f' % r,'H=1200001','T=10','L=50','tuneK=10']))

        NNsAndEfs = [ (10, [800, 400, 200, 100, 50, 30, 20, 15, 10]),
                     (5,  [30, 25, 20, 15, 10, 5, 4, 3, 2, 1]) ]
        for oneCase in NNsAndEfs:
          for ef in oneCase[1]:
            algos['SW-graph(nmslib)'].append(NmslibReuseIndex(m, 'sw-graph', 
                                            ['NN=%d' % oneCase[0], 'efConstruction=400', 'initIndexAttempts=1'], save_index,
                                            ['efSearch=%d' % ef,   'initSearchAttempts=1']))

    # END: Non-Metric Space Library (nmslib) entries

    if m == 'angular':
        # kgraph 
        kgraph_preset ={'reverse':-1, 'K':200, 'L':300, 'S':20};
        kgraph_Ps = [1,2,3,4,5,10,20,30,40,50,60,70,80,90,100]
        algos['kgraph'] = [KGraph(m, P, kgraph_preset, save_index) for P in kgraph_Ps]

        # nmslib algorithms
        MsPostsEfs=[
                (48, 2, [50,  70,  90,  120,  160,  200, 400, 600, 700, 800, 1000, 1400, 1600, 2000]),
                (32, 2, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 300, 400, 600, 700, 800, 1000, 1200, 1400,1600, 2000]),
                (20, 0, [2, 5, 10, 15, 20, 30, 40, 50, 70, 80]),
                (12, 0, [1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 80])]

        for oneCase in MsPostsEfs:
            for ef in oneCase[2]:
                algos['hnsw(nmslib)'].append(NmslibReuseIndex(m, 'hnsw', ['M=%d' % oneCase[0], 'post=%d' % oneCase[1], 'efConstruction=800'], 
                                                                         save_index,
                                                                         ['ef=%d' %ef]))

        NNsAndEfs = [ (30, [700, 650, 550, 450, 350, 275, 200, 150, 120, 80, 50, 30]),
                      (15, [80, 50, 30, 20]),
                      (3,  [120, 80, 60, 40, 20, 10, 8, 4, 2]) ]

        for oneCase in NNsAndEfs:
          for ef in oneCase[1]:
            algos['SW-graph(nmslib)'].append(NmslibReuseIndex(m, 'sw-graph', 
                                            ['NN=%d' % oneCase[0], 'efConstruction=800', 'initIndexAttempts=1'], save_index,
                                            ['efSearch=%d' % ef,   'initSearchAttempts=1']))

        # END: Non-Metric Space Library (nmslib) entries
        # RPForest only works for cosine
        algos['rpforest'] = [RPForest(leaf_size, n_trees) for n_trees in [3, 5, 10, 20, 40, 100, 200, 400] for leaf_size in [3, 5, 10, 20, 40, 100, 200, 400]]
        L = []
        x = 1
        while True:
            L.append(x)
            if x >= 1400:
                break
            x = int(math.ceil(x * 1.1))
        algos['falconn'] = [FALCONN(m, 16, l, l) for l in L]

    return algos


def get_fn(base, args):
    fn = os.path.join(base, args.dataset)

    if args.limit != -1:
        fn += '-%d' % args.limit
    if os.path.exists(fn + '.gz'):
        fn += '.gz'
    else:
        fn += '.txt'

    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)

    return fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Which dataset',  default='glove')
    parser.add_argument('--distance', help='Distance', default='angular')
    parser.add_argument('--limit', help='Limit', type=int, default=-1)
    parser.add_argument('--algo', help='run only this algorithm', default=None)
    parser.add_argument('--no_save_index', help='Do not save indices', action='store_true')

    args = parser.parse_args()

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)

    print('storing queries in', queries_fn, 'and results in', results_fn)

    if not os.path.exists(queries_fn):
        queries = get_queries(args)
        f = open(queries_fn, 'w')
        pickle.dump(queries, f)
        f.close()
    else:
        queries = pickle.load(open(queries_fn))

    print('got', len(queries), 'queries')

    algos_already_ran = set()
    if os.path.exists(results_fn):
        for line in open(results_fn):
            library, algo_name = line.strip().split('\t')[:2]
            algos_already_ran.add((library, algo_name))

    algos = get_algos(args.distance, not args.no_save_index)

    if args.algo:
        print('running only', args.algo)
        algos = {args.algo: algos[args.algo]}

    algos_flat = []

    for library in algos.keys():
        for algo in algos[library]:
            if (library, algo.name) not in algos_already_ran:
                algos_flat.append((library, algo))
                
    random.shuffle(algos_flat)

    print('order:', [a.name for l, a in algos_flat])

    for library, algo in algos_flat:
        print(algo.name, '...')
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(target=run_algo, args=(args, library, algo, results_fn))
        p.start()
        p.join()
