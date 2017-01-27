import gzip, numpy, time, os, multiprocessing, argparse, pickle, resource, random, math, yaml
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import sklearn.preprocessing
import shlex, subprocess
import json

# Set resource limits to prevent memory bombs
memory_limit = 12 * 2**30
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
if soft == resource.RLIM_INFINITY or soft >= memory_limit:
    print('resetting memory limit from', soft, 'to', memory_limit)
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))

# Nmslib specific code
# Remove old indices stored on disk
INDEX_DIR='indices'    
import shutil
if os.path.exists(INDEX_DIR):
    shutil.rmtree(INDEX_DIR)

from scipy.spatial.distance import pdist

pd = {
    'hamming': lambda a, b: pdist([a, b], metric = "hamming")[0],
    'euclidean': lambda a, b: pdist([a, b], metric = "euclidean")[0],
    'angular': lambda a, b: pdist([a, b], metric = "cosine")[0]
}

class BaseANN(object):
    def use_threads(self):
        return True
    def done(self):
        pass

import sys
sys.path.append('install/ann-filters/build/wrappers/swig/')
import locality_sensitive
class ITUFilteringDouble(BaseANN):
    def __init__(self, metric, alpha = None, beta = None, threshold = None, tau = None, kappa1 = None, kappa2 = None, m1 = None, m2 = None):
        self._loader = locality_sensitive.double_vector_loader()
        self._context = None
        self._strategy = None
        self._metric = metric
        self._alpha = alpha
        self._beta = beta
        self._threshold = threshold
        self._tau = tau
        self._kappa1 = kappa1
        self._kappa2 = kappa2
        self._m1 = m1
        self._m2 = m2
        self.name = ("ITUFilteringDouble(..., threshold = %f, ...)" % threshold)

    def fit(self, X):
        if self._metric == 'angular':
            X /= numpy.linalg.norm(X, axis=1).reshape(-1,  1)
        self._loader.add(X)
        self._context = locality_sensitive.double_vector_context(
            self._loader, self._alpha, self._beta)
        self._strategy = locality_sensitive.factories.make_double_filtering(
            self._context, self._threshold,
            locality_sensitive.filtering_configuration.from_values(
                self._kappa1, self._kappa2, self._tau, self._m1, self._m2))

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        return self._strategy.find(v, n, None)

    def use_threads(self):
        return False

class ITUHashing(BaseANN):
    def __init__(self, seed, c = 2.0, r = 2.0):
        self._loader = locality_sensitive.bit_vector_loader()
        self._context = None
        self._strategy = None
        self._c = c
        self._r = r
        self._seed = seed
        self.name = ("ITUHashing(c = %f, r = %f, seed = %u)" % (c, r, seed))

    def fit(self, X):
        locality_sensitive.set_seed(self._seed)
        for entry in X:
            locality_sensitive.hacks.add(self._loader, entry.tolist())
        self._context = locality_sensitive.bit_vector_context(
            self._loader, self._c, self._r)
        self._strategy = locality_sensitive.factories.make_hashing(
            self._context)

    def query(self, v, n):
        return locality_sensitive.hacks.find(self._strategy, n, v.tolist())

    def use_threads(self):
        return False

class Subprocess(BaseANN):
    def __raw_line(self):
        return shlex.split( \
            self.__get_program_handle().stdout.readline().strip())
    def __line(self):
        line = self.__raw_line()
        while len(line) < 1 or line[0] != "epbprtv0":
            line = self.__raw_line()
        return line[1:]

    @staticmethod
    def __quote(token):
        return "'" + str(token).replace("'", "'\\'") + "'"

    def __write(self, string):
        self.__get_program_handle().stdin.write(string + "\n")

    def __get_program_handle(self):
        if not self._program:
            self._program = subprocess.Popen(
                self._args,
                bufsize = 1, # line buffering
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                universal_newlines = True)
            for key, value in self._params.iteritems():
                self.__write("%s %s" % \
                    (Subprocess.__quote(key), Subprocess.__quote(value)))
                assert(self.__line()[0] == "ok")
            self.__write("")
            assert(self.__line()[0] == "ok")
        return self._program

    def __init__(self, args, encoder, params):
        self.name = "Subprocess(program = %s, %s)" % (args[0], str(params))
        self._program = None
        self._args = args
        self._encoder = encoder
        self._params = params

    def fit(self, X):
        for entry in X:
            self.__write(self._encoder(entry))
            assert(self.__line()[0] == "ok")
        self.__write("")
        assert(self.__line()[0] == "ok")

    def query(self, v, n):
        self.__write("%s %d" % \
            (Subprocess.__quote(self._encoder(v)), n))
        status = self.__line()
        if status[0] == "ok":
            count = int(status[1])
            results = []
            i = 0
            while i < count:
                line = self.__line()
                results.append(int(line[0]))
                i += 1
            assert(len(results) == count)
            return results
        else:
            assert(status[0] == "fail")
            return []

    def use_threads(self):
        return False
    def done(self):
        if self._program:
            self._program.terminate()

class FALCONN(BaseANN):
    def __init__(self, metric, num_bits, num_tables, num_probes = None):
        if not num_probes:
            num_probes = num_tables
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

    def use_threads(self):
        # See https://github.com/FALCONN-LIB/FALCONN/issues/6
        return False


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
            self._kgraph.save(path)

    def query(self, v, n):
        if v.dtype != numpy.float32:
            v = v.astype(numpy.float32)
        result = self._kgraph.search(numpy.array([v]), K=n, threads=1, P=self._P)
        return result[0]

class NmslibReuseIndex(BaseANN):
    @staticmethod
    def encode(d):
        return ["%s=%s" % (a, b) for (a, b) in d.iteritems()]
    def __init__(self, metric, method_name, index_param, save_index, query_param):
        self._nmslib_metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]
        self._method_name = method_name
        self._save_index = save_index
        self._index_param = NmslibReuseIndex.encode(index_param)
        self._query_param = NmslibReuseIndex.encode(query_param)
        self.name = 'Nmslib(method_name=%s, index_param=%s, query_param=%s)' % (self._method_name, self._index_param, self._query_param)
        self._index_name = os.path.join(INDEX_DIR, "nmslib_%s_%s_%s" % (self._method_name, metric, '_'.join(self._index_param)))

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
          os.makedirs(d)

    def fit(self, X):
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
        self._method_param = NmslibReuseIndex.encode(method_param)
        self.name = 'Nmslib(method_name=%s, method_param=%s)' % (self._method_name, self._method_param)

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
        metric = {'angular': 'cosine', 'euclidean': 'l2', 'hamming': 'hamming'}[self._metric]
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric=metric)
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors([v],
            return_distance = False, n_neighbors = n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors([v],
            return_distance = True, n_neighbors = n)
        return zip(list(positions[0]), list(distances[0]))

class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, metric, precision=numpy.float32):
        if metric not in ('angular', 'euclidean', 'hamming'):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
        elif metric == 'hamming' and precision != numpy.bool:
            raise NotImplementedError("BruteForceBLAS doesn't support precision %s with Hamming distances" % precision)
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
        elif self._metric == 'hamming':
            self.index = numpy.ascontiguousarray(
                map(numpy.packbits, X), dtype=numpy.uint8)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!

    def query(self, v, n):
        return map(lambda (index, _): index, self.query_with_distances(v, n))

    popcount = []
    for i in xrange(256):
      popcount.append(bin(i).count("1"))

    def query_with_distances(self, v, n):
        """Find indices of `n` most similar vectors from the index to query vector `v`."""
        if self._metric == 'hamming':
            v = numpy.packbits(v)

        # use same precision for query as for index
        v = numpy.ascontiguousarray(v, dtype = self.index.dtype)

        # HACK we ignore query length as that's a constant not affecting the final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == 'hamming':
            diff = numpy.bitwise_xor(v, self.index)
            pc = BruteForceBLAS.popcount
            den = float(len(v) * 8)
            dists = [sum([pc[part] for part in point]) / den for point in diff]
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!
        indices = numpy.argpartition(dists, n)[:n]  # partition-sort by distance, get `n` closest
        def fix(index):
            ep = self.index[index]
            ev = v
            if self._metric == "hamming":
                ep = numpy.unpackbits(ep)
                ev = numpy.unpackbits(ev)
            return (index, pd[self._metric](ep, ev))
        return map(fix, indices)

ds_loaders = {
    'float': lambda line: [float(x) for x in line.strip().split()],
    'bit': lambda line: [bool(int(x)) for x in list(line.strip())]
}

ds_printers = {
    'float': lambda X: " ".join(map(str, X)),
    'bit': lambda X: "".join(map(lambda el: "1" if el else "0", X))
}

ds_numpy_types = {
    'float': numpy.float,
    'bit': numpy.bool_
}

ds_finishers = {
    'float': lambda X: numpy.vstack(X)
}

def get_dataset(which = 'glove',
        limit = -1, random_state = 3, test_size = 10000):
    cache = 'queries/%s-%d-%d-%d.npz' % (which, test_size, limit, random_state)
    if os.path.exists(cache):
        v = numpy.load(cache)
        X_train = v['train']
        X_test = v['test']
        manifest = v['manifest'][0]
        print(manifest, X_train.shape, X_test.shape)
        return manifest, X_train, X_test
    local_fn = os.path.join('install', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    manifest = {
      'point_type': 'float'
    }
    if os.path.exists(local_fn + '.yaml'):
        y = yaml.load(open(local_fn + '.yaml'))
        if 'dataset' in y:
            manifest.update(y['dataset'])

    point_type = manifest['point_type']

    loader = None
    if not point_type in ds_loaders:
        assert False, \
            "dataset %s: unknown point type '%s'" % (which, point_type)
    else:
        loader = ds_loaders[point_type]

    X = []
    for i, line in enumerate(f):
        X.append(loader(line))
        if limit != -1 and len(X) == limit:
            break
    X = numpy.array(X, dtype = ds_numpy_types.get(point_type))

    if point_type in ds_finishers:
        X = ds_finishers[point_type](X)

    import sklearn.cross_validation

    # Here Erik is most welcome to use any other random_state
    # However, it is best to use a new random seed for each major re-evaluation,
    # so that we test on a trully bind data.
    X_train, X_test = \
      sklearn.cross_validation.train_test_split(
          X, test_size = test_size, random_state = random_state)
    print(X_train.shape, X_test.shape)

    numpy.savez(cache, manifest = [manifest], train = X_train, test = X_test)
    return manifest, X_train, X_test

def run_algo(X_train, queries, library, algo, distance, results_fn):
    try:
        t0 = time.time()
        if algo != 'bf':
            algo.fit(X_train)
        build_time = time.time() - t0
        print('Built index in', build_time)

        best_search_time = float('inf')
        best_precision = 0.0 # should be deterministic but paranoid
        for i in xrange(3): # Do multiple times to warm up page cache, use fastest
            def single_query(t):
                v, _, _ = t
                start = time.time()
                found = algo.query(v, 10)
                total = (time.time() - start)
                found = map(
                    lambda idx: (int(idx), float(pd[distance](v, X_train[idx]))),
                    list(found))
                return (total, found)
            if algo.use_threads():
                pool = multiprocessing.pool.ThreadPool()
                results = pool.map(single_query, queries)
            else:
                results = map(single_query, queries)

            total_time = sum(map(lambda (time, _): time, results))
            search_time = total_time / len(queries)
            best_search_time = min(best_search_time, search_time)

        output = {
            "library": library,
            "name": algo.name,
            "build_time": build_time,
            "best_search_time": best_search_time,
            "results": results
        }
        print(output)

        f = open(results_fn, 'a')
        f.write(json.dumps(output) + "\n")
        f.close()
    finally:
        algo.done()

def compute_distances(distance, X_train, X_test):
    print('computing max distances for queries...')

    bf = BruteForceBLAS(distance, precision = X_train.dtype)
    # Prepare queries
    bf.fit(X_train)
    queries = []
    for x in X_test:
        correct = bf.query_with_distances(x, 10)
        max_distance = max(correct, key = lambda (_, distance): distance)[1]
        queries.append((x, max_distance, correct))
        if len(queries) % 100 == 0:
            print(len(queries), '...')

    return queries

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

def BitSubprocess(args, params):
    return Subprocess(args, ds_printers["bit"], params)

from itertools import product

with open("algos.yaml", "r") as file:
    algorithms = yaml.load(file)

def handle_args(args):
    if isinstance(args, list):
        args = map(lambda el: el if isinstance(el, list) else [el], args)
        return map(list, product(*args))
    elif isinstance(args, dict):
        flat = []
        for k, v in args.iteritems():
            if isinstance(v, list):
                flat.append(map(lambda el: (k, el), v))
            else:
                flat.append([(k, v)])
        return map(dict, product(*flat))
    else:
        raise TypeError("No args handling exists for %s" % type(args).__name__)

point_type = "float"
distance_metric = "euclidean"

def get_definitions(point_type, distance_metric):
    algorithm_definitions = {}
    if "any" in algorithms[point_type]:
        algorithm_definitions.update(algorithms[point_type]["any"])
    algorithm_definitions.update(algorithms[point_type][distance_metric])

    algos = {}
    for (name, algo) in algorithm_definitions.iteritems():
        assert "constructor" in algo, \
                "group %s does not specify a constructor" % name
        cn = algo["constructor"]
        assert cn in globals(), \
                "group %s specifies the nonexistent constructor %s" % (name, cn)
        constructor = globals()[cn]

        algos[name] = []

        base_args = []
        vs = {
            "@metric": distance_metric
        }
        if "base-args" in algo:
            base_args = map(lambda arg: arg \
                            if not isinstance(arg, str) or \
                            not arg in vs else vs[arg],
                            algo["base-args"])

        for run_group in algo["run-groups"].values():
            if "arg-groups" in run_group:
                groups = []
                for arg_group in run_group["arg-groups"]:
                    if isinstance(arg_group, dict):
                        # Dictionaries need to be expanded into lists in order
                        # for the subsequent call to handle_args to do the
                        # right thing
                        groups.append(handle_args(arg_group))
                    else:
                        groups.append(arg_group)
                args = handle_args(groups)
            elif "args" in run_group:
                args = handle_args(run_group["args"])
            else:
                assert False, "? what? %s" % run_group

            for arg_group in args:
                obj = None
                try:
                    aargs = []
                    aargs.extend(base_args)
                    if isinstance(arg_group, list):
                        aargs.extend(arg_group)
                    else:
                        aargs.append(arg_group)
                    obj = constructor(*aargs)
                    algos[name].append(obj)
                except Exception as e:
                    print e
                    pass
    return algos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Which dataset',  default='glove')
    parser.add_argument('--distance', help='Distance', default='angular')
    parser.add_argument('--limit', help='Limit', type=int, default=-1)
    parser.add_argument('--algo', help='run only this algorithm', default=None)
    parser.add_argument('--sub-algo', help='run only the named algorithm instance', default=None)
    parser.add_argument('--no_save_index', help='Do not save indices', action='store_true')

    args = parser.parse_args()

    manifest, X_train, X_test = get_dataset(args.dataset, args.limit)

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)

    print('storing queries in', queries_fn, 'and results in', results_fn)

    if not os.path.exists(queries_fn):
        queries = compute_distances(args.distance, X_train, X_test)
        f = open(queries_fn, 'w')
        pickle.dump(queries, f)
        f.close()
    else:
        queries = pickle.load(open(queries_fn))

    print('got', len(queries), 'queries')

    algos_already_ran = set()
    if os.path.exists(results_fn):
        for line in open(results_fn):
            run = json.loads(line)
            algos_already_ran.add((run["library"], run["name"]))

    point_type = manifest['point_type']
    algos = get_definitions(point_type, args.distance)

    if args.algo:
        print('running only', args.algo)
        algos = {args.algo: algos[args.algo]}
        if args.sub_algo:
            algos[args.algo] = \
              [algo for algo in algos[args.algo] if algo.name == args.sub_algo]

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
        p = multiprocessing.Process(
            target = run_algo,
            args =
                (X_train, queries, library, algo, args.distance, results_fn))
        p.start()
        p.join()
