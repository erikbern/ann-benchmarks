import h5py
import numpy
import os
import random
import sys
import urllib.request


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urllib.request.urlretrieve(src, dst)


def get_dataset_fn(dataset):
    return os.path.join('data', '%s.hdf5' % dataset)


def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    url = 'http://vectors.erikbern.com/%s.hdf5' % which
    download(url, hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn)
    return hdf5_f


# Everything below this line is related to creating datasets
# You probably never need to do this at home, just rely on the prepared datasets at http://vectors.erikbern.com

def write_output(train, test, fn, distance, count=100):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    n = 0
    f = h5py.File(fn, 'w')
    f.attrs['distance'] = distance
    print('train size: %d * %d' % train.shape)
    print('test size: %d * %d' % test.shape)
    f.create_dataset('train', (len(train), len(train[0])), dtype='f')[:] = train
    f.create_dataset('test', (len(test), len(test[0])), dtype='f')[:] = test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    bf = BruteForceBLAS(distance, precision=numpy.float32)
    bf.fit(train)
    queries = []
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print('%d/%d...' % (i, test.shape[0]))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()


def glove(out_fn, d):
    import sklearn.model_selection
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        print('splitting output...')
        X_train, X_test = sklearn.model_selection.train_test_split(X, test_size=10000, random_state=1)
        print('writing output...')
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, 'angular')


def _load_texmex_vectors(f):
    import struct

    vs = []
    while True:
        b = f.read(4)
        if not b:
            break
        dim = struct.unpack('i', b)[0]
        vec = struct.unpack('f' * dim, f.read(dim*4))
        vs.append(vec)
    return numpy.array(vs)


def sift(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _load_texmex_vectors(t.extractfile(t.getmember('sift/sift_base.fvecs')))
        test = _load_texmex_vectors(t.extractfile(t.getmember('sift/sift_query.fvecs')))
        write_output(train, test, out_fn, 'euclidean')


def gist(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz'
    fn = os.path.join('data', 'gist.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _load_texmex_vectors(t.extractfile(t.getmember('gist/gist_base.fvecs')))
        test = _load_texmex_vectors(t.extractfile(t.getmember('gist/gist_query.fvecs')))
        write_output(train, test, out_fn, 'euclidean')


def _load_mnist_vectors(fn):
    import gzip
    import struct

    print('parsing vectors in %s...' % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d")
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn):
    download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-train.gz')
    download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist-test.gz')
    train = _load_mnist_vectors('mnist-train.gz')
    test = _load_mnist_vectors('mnist-test.gz')
    write_output(train, test, out_fn, 'euclidean')



def fashion_mnist(out_fn):
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz', 'fashion-mnist-train.gz')
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz', 'fashion-mnist-test.gz')
    train = _load_mnist_vectors('fashion-mnist-train.gz')
    test = _load_mnist_vectors('fashion-mnist-test.gz')
    write_output(train, test, out_fn, 'euclidean')

def transform_bag_of_words(filename, n_dimensions, out_fn):
    import gzip
    import sklearn.model_selection
    from scipy.sparse import lil_matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn import random_projection
    with gzip.open(filename, 'rb') as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:] # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(n_components = n_dimensions).fit_transform(B)
        X_train, X_test = sklearn.model_selection.train_test_split(C, test_size=10000, random_state=1)
        print('writing output...')
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, 'angular')

def nytimes(out_fn, n_dimensions):
    download('https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz', 'nytimes.txt.gz')
    transform_bag_of_words('nytimes.txt.gz', n_dimensions, out_fn)


def random(out_fn, n_dims, n_samples, centers, distance):
    import sklearn.model_selection
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1)
    X_train, X_test = sklearn.model_selection.train_test_split(X, test_size=0.1, random_state=1)
    write_output(X_train, X_test, out_fn, distance)

DATASETS = {
    'fashion-mnist-784-euclidean': fashion_mnist,
    'gist-960-euclidean': gist,
    'glove-25-angular': lambda out_fn: glove(out_fn, 25),
    'glove-50-angular': lambda out_fn: glove(out_fn, 50),
    'glove-100-angular': lambda out_fn: glove(out_fn, 100),
    'glove-200-angular': lambda out_fn: glove(out_fn, 200),
    'mnist-784-euclidean': mnist,
    'random-xs-10-euclidean': lambda out_fn: random(out_fn, 10, 10000, 100, 'euclidean'),
    'random-s-40-euclidean': lambda out_fn: random(out_fn, 40, 100000, 1000, 'euclidean'),
    'random-xs-10-angular': lambda out_fn: random(out_fn, 10, 10000, 100, 'angular'),
    'random-s-40-angular': lambda out_fn: random(out_fn, 40, 100000, 1000, 'angular'),
    'sift-128-euclidean': sift,
    'nytimes-256-angular': lambda out_fn: nytimes(out_fn, 256),
}
