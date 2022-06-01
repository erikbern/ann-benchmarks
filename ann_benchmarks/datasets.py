from copyreg import pickle
import h5py
import numpy
import os
import random

from urllib.request import urlopen
from urllib.request import urlretrieve

from ann_benchmarks.distance import dataset_transform
import urllib.parse


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists('data'):
        try:
            os.mkdir('data')
        except FileExistsError:
            pass # fixes race condition
    return os.path.join('data', '%s.hdf5' % dataset)


def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        if 'dbpedia' in which:
             url = 'https://s3.us-east-1.amazonaws.com/benchmarks.redislabs/vecsim/dbpedia/dbpedia-768.hdf5'
        elif 'amazon-reviews' in which:
             url = 'https://s3.us-east-1.amazonaws.com/benchmarks.redislabs/vecsim/amazon_reviews/amazon-reviews-384.hdf5'
        elif 'hybrid' in which:
            url = 'https://s3.us-east-1.amazonaws.com/benchmarks.redislabs/vecsim/hybrid_datasets/%s.hdf5' % urllib.parse.quote(which)
        elif 'Text-to-Image' in which:
            url = 'https://s3.us-east-1.amazonaws.com/benchmarks.redislabs/vecsim/big_ann/%s.hdf5' % urllib.parse.quote(which)
        else:    
            url = 'http://ann-benchmarks.com/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn, 'r')

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(hdf5_f['train'][0])

    return hdf5_f, dimension

# Everything below this line is related to creating datasets
# You probably never need to do this at home,
# just rely on the prepared datasets at http://ann-benchmarks.com


def write_output(train, test, fn, distance, point_type='float', count=100):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    n = 0
    f = h5py.File(fn, 'w')
    f.attrs['type'] = 'dense'
    f.attrs['distance'] = distance
    f.attrs['dimension'] = len(train[0])
    f.attrs['point_type'] = point_type
    print('train size: %9d * %4d' % train.shape)
    print('test size:  %9d * %4d' % test.shape)
    f.create_dataset('train', (len(train), len(
        train[0])), dtype=train.dtype)[:] = train
    f.create_dataset('test', (len(test), len(
        test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    bf = BruteForceBLAS(distance, precision=train.dtype)

    bf.fit(train)
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print('%d/%d...' % (i, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()

"""
param: train and test are arrays of arrays of indices.
"""
def write_sparse_output(train, test, fn, distance, dimension, count=100):
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    f = h5py.File(fn, 'w')
    f.attrs['type'] = 'sparse'
    f.attrs['distance'] = distance
    f.attrs['dimension'] = dimension
    f.attrs['point_type'] = 'bit'
    print('train size: %9d * %4d' % (train.shape[0], dimension))
    print('test size:  %9d * %4d' % (test.shape[0], dimension))

    # We ensure the sets are sorted
    train = numpy.array(list(map(sorted, train)))
    test = numpy.array(list(map(sorted, test)))

    flat_train = numpy.hstack(train.flatten())
    flat_test = numpy.hstack(test.flatten())

    f.create_dataset('train', (len(flat_train),), dtype=flat_train.dtype)[:] = flat_train
    f.create_dataset('test', (len(flat_test),), dtype=flat_test.dtype)[:] = flat_test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')

    f.create_dataset('size_test', (len(test),), dtype='i')[:] = list(map(len, test))
    f.create_dataset('size_train', (len(train),), dtype='i')[:] = list(map(len, train))

    bf = BruteForceBLAS(distance, precision=train.dtype)
    bf.fit(train)
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print('%d/%d...' % (i, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    f.close()

def train_test_split(X, test_size=10000, dimension=None):
    import sklearn.model_selection
    if dimension == None:
        dimension = X.shape[1]
    print('Splitting %d*%d into train/test' % (X.shape[0], dimension))
    return sklearn.model_selection.train_test_split(
        X, test_size=test_size, random_state=1)


def glove(out_fn, d):
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
        X_train, X_test = train_test_split(X)
        write_output(numpy.array(X_train), numpy.array(
            X_test), out_fn, 'angular')


def _load_texmex_vectors(f, n, k):
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack('f' * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct
    m = t.getmember(fn)
    f = t.extractfile(m)
    k, = struct.unpack('i', f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        test = _get_irisa_matrix(t, 'sift/sift_query.fvecs')
        write_output(train, test, out_fn, 'euclidean')


def gist(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz'
    fn = os.path.join('data', 'gist.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'gist/gist_base.fvecs')
        test = _get_irisa_matrix(t, 'gist/gist_query.fvecs')
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

    dimensions = [struct.unpack("!I", f.read(4))[0]
                  for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0]
                        for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn):
    download(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-train.gz')  # noqa
    download(
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist-test.gz')  # noqa
    train = _load_mnist_vectors('mnist-train.gz')
    test = _load_mnist_vectors('mnist-test.gz')
    write_output(train, test, out_fn, 'euclidean')


def fashion_mnist(out_fn):
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',  # noqa
             'fashion-mnist-train.gz')
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',  # noqa
             'fashion-mnist-test.gz')
    train = _load_mnist_vectors('fashion-mnist-train.gz')
    test = _load_mnist_vectors('fashion-mnist-test.gz')
    write_output(train, test, out_fn, 'euclidean')

# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn):
    yadisk_key = 'https://yadi.sk/d/11eDCm7Dsn9GA'
    response = urlopen('https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
        + yadisk_key + '&path=/deep10M.fvecs')
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(',')[0][9:-1]
    filename = os.path.join('data', 'deep-image.fvecs')
    download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = numpy.fromfile(filename, dtype=numpy.float32)
    dim = fv.view(numpy.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv)
    write_output(X_train, X_test, out_fn, 'angular')

def transform_bag_of_words(filename, n_dimensions, out_fn):
    import gzip
    from scipy.sparse import lil_matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn import random_projection
    with gzip.open(filename, 'rb') as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(
            n_components=n_dimensions).fit_transform(B)
        X_train, X_test = train_test_split(C)
        write_output(numpy.array(X_train), numpy.array(
            X_test), out_fn, 'angular')


def nytimes(out_fn, n_dimensions):
    fn = 'nytimes_%s.txt.gz' % n_dimensions
    download('https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz', fn)  # noqa
    transform_bag_of_words(fn, n_dimensions, out_fn)


def random_float(out_fn, n_dims, n_samples, centers, distance):
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=n_dims,
        centers=centers, random_state=1)
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def random_bitstring(out_fn, n_dims, n_samples, n_queries):
    import sklearn.datasets

    Y, _ = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=n_dims,
        centers=n_queries, random_state=1)
    X = numpy.zeros((n_samples, n_dims), dtype=numpy.bool)
    for i, vec in enumerate(Y):
        X[i] = numpy.array([v > 0 for v in vec], dtype=numpy.bool)

    X_train, X_test = train_test_split(X, test_size=n_queries)
    write_output(X_train, X_test, out_fn, 'hamming', 'bit')


def word2bits(out_fn, path, fn):
    import tarfile
    local_fn = fn + '.tar.gz'
    url = 'http://web.stanford.edu/~maxlam/word_vectors/compressed/%s/%s.tar.gz' % (  # noqa
        path, fn)
    download(url, local_fn)
    print('parsing vectors in %s...' % local_fn)
    with tarfile.open(local_fn, 'r:gz') as t:
        f = t.extractfile(fn)
        n_words, k = [int(z) for z in next(f).strip().split()]
        X = numpy.zeros((n_words, k), dtype=numpy.bool)
        for i in range(n_words):
            X[i] = numpy.array([float(z) > 0 for z in next(
                f).strip().split()[1:]], dtype=numpy.bool)

        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, 'hamming', 'bit')


def sift_hamming(out_fn, fn):
    import tarfile
    local_fn = fn + '.tar.gz'
    url = 'http://sss.projects.itu.dk/ann-benchmarks/datasets/%s.tar.gz' % fn
    download(url, local_fn)
    print('parsing vectors in %s...' % local_fn)
    with tarfile.open(local_fn, 'r:gz') as t:
        f = t.extractfile(fn)
        lines = f.readlines()
        X = numpy.zeros((len(lines), 256), dtype=numpy.bool)
        for i, line in enumerate(lines):
            X[i] = numpy.array(
                [int(x) > 0 for x in line.decode().strip()], dtype=numpy.bool)
        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, 'hamming', 'bit')

def kosarak(out_fn):
    import gzip
    local_fn = 'kosarak.dat.gz'
    # only consider sets with at least min_elements many elements
    min_elements = 20
    url = 'http://fimi.uantwerpen.be/data/%s' % local_fn
    download(url, local_fn)

    X = []
    dimension = 0
    with gzip.open('kosarak.dat.gz', 'r') as f:
        content = f.readlines()
        # preprocess data to find sets with more than 20 elements
        # keep track of used ids for reenumeration
        for line in content:
            if len(line.split()) >= min_elements:
                X.append(list(map(int, line.split())))
                dimension = max(dimension, max(X[-1]) + 1)

    X_train, X_test = train_test_split(numpy.array(X), test_size=500, dimension=dimension)
    write_sparse_output(X_train, X_test, out_fn, 'jaccard', dimension)

def random_jaccard(out_fn, n=10000, size=50, universe=80):
    random.seed(1)
    l = list(range(universe))
    X = []
    for i in range(n):
        X.append(random.sample(l, size))

    X_train, X_test = train_test_split(numpy.array(X), test_size=100, dimension=universe)
    write_sparse_output(X_train, X_test, out_fn, 'jaccard', universe)



def lastfm(out_fn, n_dimensions, test_size=50000):
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)
    from implicit.datasets.lastfm import get_lastfm
    from implicit.approximate_als import augment_inner_product_matrix
    import implicit

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(
        play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = numpy.append(model.user_factors,
                                numpy.zeros((model.user_factors.shape[0], 1)),
                                axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, user_factors, out_fn, 'angular')

def parse_dbpedia_data(source_file, max_docs: int):
    import re
    """
    Parses the input file of abstracts and returns an iterable
    :param max_docs: maximum number of input documents to process; -1 for no limit
    :param source_file: input file
    :return: yields document by document to the consumer
    """
    global VERBOSE
    count = 0
    max_tokens = 0

    if -1 < max_docs < 50:
        VERBOSE = True

    percent = 0.1
    bulk_size = (percent / 100) * max_docs

    print(f"bulk_size={bulk_size}")

    if bulk_size <= 0:
        bulk_size = 1000

    for line in source_file:
        line = line.decode("utf-8")

        # skip commented out lines
        comment_regex = '^#'
        if re.search(comment_regex, line):
            continue

        token_size = len(line.split())
        if token_size > max_tokens:
            max_tokens = token_size

        # skip lines with 20 tokens or less, because they tend to contain noise
        # (this may vary in your dataset)
        if token_size <= 20:
            continue

        first_url_regex = '^<([^\>]+)>\s*'

        x = re.search(first_url_regex, line)
        if x:
            url = x.group(1)
            # also remove the url from the string
            line = re.sub(first_url_regex, '', line)
        else:
            url = ''

        # remove the second url from the string: we don't need to capture it, because it is repetitive across
        # all abstracts
        second_url_regex = '^<[^\>]+>\s*'
        line = re.sub(second_url_regex, '', line)

        # remove some strange line ending, that occurs in many abstracts
        language_at_ending_regex = '@en \.\n$'
        line = re.sub(language_at_ending_regex, '', line)

        # form the input object for this abstract
        doc = {
            "_text_": line,
            "url": url,
            "id": count+1
        }

        yield doc
        count += 1

        if count % bulk_size == 0:
            print(f"Processed {count} documents", end="\r")

        if count == max_docs:
            break

    source_file.close()
    print("Maximum tokens observed per abstract: {}".format(max_tokens))

def dbpedia(out_fn):
    import bz2
    from sentence_transformers import SentenceTransformer
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    local_fn = "long_abstracts_en.ttl.bz2"
    url = "http://downloads.dbpedia.org/2016-10/core-i18n/en/long_abstracts_en.ttl.bz2"
    download(url, local_fn)
    source_file = bz2.BZ2File(local_fn, "r")
    docs_iter = parse_dbpedia_data(source_file=source_file, max_docs=1000000)
    text = []
    for doc in docs_iter:
        text.append(doc['_text_'])
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    model.to(device)
    sentence_embeddings = model.encode(text, show_progress_bar=True)
    write_output(sentence_embeddings, sentence_embeddings[:10000], out_fn, 'angular')


def amazon_reviews(out_fn):
    import os
    import math
    import pickle
    import numpy as np
    subsets = ['Wireless_v1_00', 'Watches_v1_00', 'Video_Games_v1_00', 'Video_DVD_v1_00', 'Video_v1_00', 'Toys_v1_00', 'Tools_v1_00', 'Sports_v1_00', 'Software_v1_00', 'Shoes_v1_00', 'Pet_Products_v1_00', 'Personal_Care_Appliances_v1_00', 'PC_v1_00', 'Outdoors_v1_00', 'Office_Products_v1_00', 'Musical_Instruments_v1_00', 'Music_v1_00', 'Mobile_Electronics_v1_00', 'Mobile_Apps_v1_00', 'Major_Appliances_v1_00', 'Luggage_v1_00', 'Lawn_and_Garden_v1_00', 'Kitchen_v1_00', 'Jewelry_v1_00', 'Home_Improvement_v1_00', 'Home_Entertainment_v1_00', 'Home_v1_00', 'Health_Personal_Care_v1_00', 'Grocery_v1_00', 'Gift_Card_v1_00', 'Furniture_v1_00', 'Electronics_v1_00', 'Digital_Video_Games_v1_00', 'Digital_Video_Download_v1_00', 'Digital_Software_v1_00', 'Digital_Music_Purchase_v1_00', 'Digital_Ebook_Purchase_v1_00', 'Camera_v1_00', 'Books_v1_00', 'Beauty_v1_00', 'Baby_v1_00', 'Automotive_v1_00', 'Apparel_v1_00', 'Digital_Ebook_Purchase_v1_01', 'Books_v1_01', 'Books_v1_02']
    train_set = None
    test_set = None
    for i, subset in enumerate(subsets):
        url = f'https://s3.us-east-1.amazonaws.com/benchmarks.redislabs/vecsim/amazon_reviews/{subset}_embeddings'
        local_fn = f'{subset}_embeddings'
        download(url, local_fn)
        subset_embeddings = pickle.load(open(local_fn, "rb"))
        if i==0:
            train_set = subset_embeddings
            test_set = subset_embeddings[:math.ceil(10000/len(subsets))]
        else:
            train_set = np.append(train_set, subset_embeddings, axis =0)
            test_set = np.append(test_set, subset_embeddings[:math.ceil(10000/len(subsets))], axis=0)
        print(subset_embeddings.shape)
        print(train_set.shape)
        print(test_set.shape)
        os.remove(local_fn)
    write_output(train_set, test_set[:10000], out_fn, 'angular')


DATASETS = {
    'deep-image-96-angular': deep_image,
    'fashion-mnist-784-euclidean': fashion_mnist,
    'gist-960-euclidean': gist,
    'glove-25-angular': lambda out_fn: glove(out_fn, 25),
    'glove-50-angular': lambda out_fn: glove(out_fn, 50),
    'glove-100-angular': lambda out_fn: glove(out_fn, 100),
    'glove-200-angular': lambda out_fn: glove(out_fn, 200),
    'mnist-784-euclidean': mnist,
    'random-xs-20-euclidean': lambda out_fn: random_float(out_fn, 20, 10000, 100,
                                                    'euclidean'),
    'random-s-100-euclidean': lambda out_fn: random_float(out_fn, 100, 100000, 1000,
                                                    'euclidean'),
    'random-xs-20-angular': lambda out_fn: random_float(out_fn, 20, 10000, 100,
                                                  'angular'),
    'random-s-100-angular': lambda out_fn: random_float(out_fn, 100, 100000, 1000,
                                                  'angular'),
    'random-xs-16-hamming': lambda out_fn: random_bitstring(out_fn, 16, 10000,
                                                            100),
    'random-s-128-hamming': lambda out_fn: random_bitstring(out_fn, 128,
                                                            50000, 1000),
    'random-l-256-hamming': lambda out_fn: random_bitstring(out_fn, 256,
                                                            100000, 1000),
    'random-s-jaccard': lambda out_fn: random_jaccard(out_fn, n=10000,
                                                       size=20, universe=40),
    'random-l-jaccard': lambda out_fn: random_jaccard(out_fn, n=100000,
                                                       size=70, universe=100),
    'sift-128-euclidean': sift,
    'nytimes-256-angular': lambda out_fn: nytimes(out_fn, 256),
    'nytimes-16-angular': lambda out_fn: nytimes(out_fn, 16),
    'word2bits-800-hamming': lambda out_fn: word2bits(
        out_fn, '400K',
        'w2b_bitlevel1_size800_vocab400K'),
    'lastfm-64-dot': lambda out_fn: lastfm(out_fn, 64),
    'sift-256-hamming': lambda out_fn: sift_hamming(
        out_fn, 'sift.hamming.256'),
    'kosarak-jaccard': lambda out_fn: kosarak(out_fn),
    'dbpedia-768' : lambda out_fn: dbpedia(out_fn),
    'amazon-reviews-384': lambda out_fn: amazon_reviews(out_fn),
}




big_ann_datasets = [f'Text-to-Image-{x}' for x in ['10M', '20M', '30M', '40M', '50M', '60M', '70M', '80M', '90M', '100M']]
for dataset in big_ann_datasets:
     DATASETS[dataset] = lambda fn: ()


hybrid_datasets = ['glove-200-angular', 'gist-960-euclidean', 'deep-image-96-angular', 'fashion-mnist-784-euclidean']
hybrid_datasets.extend(big_ann_datasets)
percentiles= ['0.5', '1', '2', '5', '10', '20', '50']
for dataset in hybrid_datasets:
    for percentile in percentiles:
        DATASETS[f'{dataset}-hybrid-{percentile}'] = lambda fn: ()

