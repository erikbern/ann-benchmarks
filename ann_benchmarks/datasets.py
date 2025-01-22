import os
import random
import tarfile
from urllib.request import build_opener, install_opener, urlopen, urlretrieve
import traceback

import h5py
import numpy
from typing import Any, Callable, Dict, Tuple

# Needed for Cloudflare's firewall
opener = build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
install_opener(opener)


def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str) -> Tuple[h5py.File, int]:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download(dataset_url, hdf5_filename)
    except:
        traceback.print_exc()
        print(f"Cannot download {dataset_url}")
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name](hdf5_filename)

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension


def write_output(train: numpy.ndarray, test: numpy.ndarray, fn: str, distance: str, point_type: str = "float", count: int = 100) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """
    from ann_benchmarks.algorithms.bruteforce.module import BruteForceBLAS

    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train[0])
        f.attrs["point_type"] = point_type
        print(f"train size: {train.shape[0]} * {train.shape[1]}")
        print(f"test size:  {test.shape[0]} * {test.shape[1]}")
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=train.dtype)
        bf.fit(train)

        for i, x in enumerate(test):
            if i % 1000 == 0:
                print(f"{i}/{len(test)}...")

            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


"""
param: train and test are arrays of arrays of indices.
"""


def write_sparse_output(train: numpy.ndarray, test: numpy.ndarray, fn: str, distance: str, dimension: int, count: int = 100) -> None:
    """
    Writes the provided sparse training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        train (numpy.ndarray): The sparse training data.
        test (numpy.ndarray): The sparse testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        dimension (int): The dimensionality of the data.
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """
    from ann_benchmarks.algorithms.bruteforce.module import BruteForceBLAS

    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "sparse"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = dimension
        f.attrs["point_type"] = "bit"
        print(f"train size: {train.shape[0]} * {dimension}")
        print(f"test size:  {test.shape[0]} * {dimension}")

        # Ensure the sets are sorted
        train = numpy.array([sorted(t) for t in train])
        test = numpy.array([sorted(t) for t in test])

        # Flatten and write train and test sets
        flat_train = numpy.concatenate(train)
        flat_test = numpy.concatenate(test)
        f.create_dataset("train", data=flat_train)
        f.create_dataset("test", data=flat_test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Write sizes of train and test sets
        f.create_dataset("size_train", data=[len(t) for t in train])
        f.create_dataset("size_test", data=[len(t) for t in test])

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=flat_train.dtype)
        bf.fit(train)

        for i, x in enumerate(test):
            if i % 1000 == 0:
                print(f"{i}/{len(test)}...")
            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


def train_test_split(X: numpy.ndarray, test_size: int = 10000, dimension: int = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Splits the provided dataset into a training set and a testing set.
    
    Args:
        X (numpy.ndarray): The dataset to split.
        test_size (int, optional): The number of samples to include in the test set. 
            Defaults to 10000.
        dimension (int, optional): The dimensionality of the data. If not provided, 
            it will be inferred from the second dimension of X. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the training set and the testing set.
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    dimension = dimension if not None else X.shape[1]
    print(f"Splitting {X.shape[0]}*{dimension} into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)


def glove(out_fn: str, d: int) -> None:
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X_train, X_test = train_test_split(X)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


def _load_texmex_vectors(f: Any, n: int, k: int) -> numpy.ndarray:
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> numpy.ndarray:
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", "sift.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def gist(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def _load_mnist_vectors(fn: str) -> numpy.ndarray:
    import gzip
    import struct

    print("parsing vectors in %s..." % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
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


def mnist(out_fn: str) -> None:
    download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "mnist-train.gz")  # noqa
    download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "mnist-test.gz")  # noqa
    train = _load_mnist_vectors("mnist-train.gz")
    test = _load_mnist_vectors("mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn: str) -> None:
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-train.gz",
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-test.gz",
    )
    train = _load_mnist_vectors("fashion-mnist-train.gz")
    test = _load_mnist_vectors("fashion-mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn: str) -> None:
    yadisk_key = "https://yadi.sk/d/11eDCm7Dsn9GA"
    response = urlopen(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        + yadisk_key
        + "&path=/deep10M.fvecs"
    )
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(",")[0][9:-1]
    filename = os.path.join("data", "deep-image.fvecs")
    download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = numpy.fromfile(filename, dtype=numpy.float32)
    dim = fv.view(numpy.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv)
    write_output(X_train, X_test, out_fn, "angular")


def transform_bag_of_words(filename: str, n_dimensions: int, out_fn: str) -> None:
    import gzip

    from scipy.sparse import lil_matrix
    from sklearn import random_projection
    from sklearn.feature_extraction.text import TfidfTransformer

    with gzip.open(filename, "rb") as f:
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
        C = random_projection.GaussianRandomProjection(n_components=n_dimensions).fit_transform(B)
        X_train, X_test = train_test_split(C)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


def nytimes(out_fn: str, n_dimensions: int) -> None:
    fn = "nytimes_%s.txt.gz" % n_dimensions
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz", fn
    )  # noqa
    transform_bag_of_words(fn, n_dimensions, out_fn)


def random_float(out_fn: str, n_dims: int, n_samples: int, centers: int, distance: str) -> None:
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1)
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def random_bitstring(out_fn: str, n_dims: int, n_samples: int, n_queries: int) -> None:
    import sklearn.datasets

    Y, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=n_queries, random_state=1)
    X = numpy.zeros((n_samples, n_dims), dtype=numpy.bool_)
    for i, vec in enumerate(Y):
        X[i] = numpy.array([v > 0 for v in vec], dtype=numpy.bool_)

    X_train, X_test = train_test_split(X, test_size=n_queries)
    write_output(X_train, X_test, out_fn, "hamming", "bit")


def sift_hamming(out_fn: str, fn: str) -> None:
    import tarfile

    local_fn = fn + ".tar.gz"
    url = "http://web.stanford.edu/~maxlam/word_vectors/compressed/%s/%s.tar.gz" % (path, fn)  # noqa
    download(url, local_fn)
    print("parsing vectors in %s..." % local_fn)
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        n_words, k = [int(z) for z in next(f).strip().split()]
        X = numpy.zeros((n_words, k), dtype=numpy.bool_)
        for i in range(n_words):
            X[i] = numpy.array([float(z) > 0 for z in next(f).strip().split()[1:]], dtype=numpy.bool_)

        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def sift_hamming(out_fn: str, fn: str) -> None:
    import tarfile

    local_fn = fn + ".tar.gz"
    url = "http://sss.projects.itu.dk/ann-benchmarks/datasets/%s.tar.gz" % fn
    download(url, local_fn)
    print("parsing vectors in %s..." % local_fn)
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        lines = f.readlines()
        X = numpy.zeros((len(lines), 256), dtype=numpy.bool_)
        for i, line in enumerate(lines):
            X[i] = numpy.array([int(x) > 0 for x in line.decode().strip()], dtype=numpy.bool_)
        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def kosarak(out_fn: str) -> None:
    import gzip

    local_fn = "kosarak.dat.gz"
    # only consider sets with at least min_elements many elements
    min_elements = 20
    url = "http://fimi.uantwerpen.be/data/%s" % local_fn
    download(url, local_fn)

    X = []
    dimension = 0
    with gzip.open("kosarak.dat.gz", "r") as f:
        content = f.readlines()
        # preprocess data to find sets with more than 20 elements
        # keep track of used ids for reenumeration
        for line in content:
            if len(line.split()) >= min_elements:
                X.append(list(map(int, line.split())))
                dimension = max(dimension, max(X[-1]) + 1)

    X_train, X_test = train_test_split(numpy.array(X), test_size=500, dimension=dimension)
    write_sparse_output(X_train, X_test, out_fn, "jaccard", dimension)


def random_jaccard(out_fn: str, n: int = 10000, size: int = 50, universe: int = 80) -> None:
    random.seed(1)
    l = list(range(universe))
    X = []
    for i in range(n):
        X.append(random.sample(l, size))

    X_train, X_test = train_test_split(numpy.array(X), test_size=100, dimension=universe)
    write_sparse_output(X_train, X_test, out_fn, "jaccard", universe)


def lastfm(out_fn: str, n_dimensions: int, test_size: int = 50000) -> None:
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
    import implicit
    from implicit.approximate_als import augment_inner_product_matrix
    from implicit.datasets.lastfm import get_lastfm

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = numpy.append(model.user_factors, numpy.zeros((model.user_factors.shape[0], 1)), axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, user_factors, out_fn, "angular")


def movielens(fn: str, ratings_file: str, out_fn: str, separator: str = "::", ignore_header: bool = False) -> None:
    import zipfile

    url = "http://files.grouplens.org/datasets/movielens/%s" % fn

    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        file = z.open(ratings_file)
        if ignore_header:
            file.readline()

        print("preparing %s" % out_fn)

        users = {}
        X = []
        dimension = 0
        for line in file:
            el = line.decode("UTF-8").split(separator)

            userId = el[0]
            itemId = int(el[1])
            rating = float(el[2])

            if rating < 3:  # We only keep ratings >= 3
                continue

            if userId not in users:
                users[userId] = len(users)
                X.append([])

            X[users[userId]].append(itemId)
            dimension = max(dimension, itemId + 1)

        X_train, X_test = train_test_split(numpy.array(X), test_size=500, dimension=dimension)
        write_sparse_output(X_train, X_test, out_fn, "jaccard", dimension)


def movielens1m(out_fn: str) -> None:
    movielens("ml-1m.zip", "ml-1m/ratings.dat", out_fn)


def movielens10m(out_fn: str) -> None:
    movielens("ml-10m.zip", "ml-10M100K/ratings.dat", out_fn)


def movielens20m(out_fn: str) -> None:
    movielens("ml-20m.zip", "ml-20m/ratings.csv", out_fn, ",", True)

def dbpedia_entities_openai_1M(out_fn, n = None):
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset
    import numpy as np

    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")
    if n is not None and n >= 100_000:
        data = data.select(range(n))

    embeddings = data.to_pandas()['openai'].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    X_train, X_test = train_test_split(embeddings, test_size=10_000, random_state=42)

    write_output(X_train, X_test, out_fn, "angular")

def coco(out_fn: str, kind: str):
    assert kind in ('t2i', 'i2i')

    local_fn = "coco-clip-b16-512-features.hdf5"
    url = "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/%s" % local_fn
    download(url, local_fn)

    with h5py.File(local_fn, "r") as f:
        img_X = f['img_feats'][:]

        X_train, X_test = train_test_split(img_X, test_size=10_000)

        if kind == 't2i':
            # there are 5 captions per image, take the first one
            txt_X = f['txt_feats'][::5]
            _, X_test = train_test_split(txt_X, test_size=10_000)

    write_output(X_train, X_test, out_fn, "angular")


DATASETS: Dict[str, Callable[[str], None]] = {
    "deep-image-96-angular": deep_image,
    "fashion-mnist-784-euclidean": fashion_mnist,
    "gist-960-euclidean": gist,
    "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "mnist-784-euclidean": mnist,
    "random-xs-20-euclidean": lambda out_fn: random_float(out_fn, 20, 10000, 100, "euclidean"),
    "random-s-100-euclidean": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "euclidean"),
    "random-xs-20-angular": lambda out_fn: random_float(out_fn, 20, 10000, 100, "angular"),
    "random-s-100-angular": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "angular"),
    "random-xs-16-hamming": lambda out_fn: random_bitstring(out_fn, 16, 10000, 100),
    "random-s-128-hamming": lambda out_fn: random_bitstring(out_fn, 128, 50000, 1000),
    "random-l-256-hamming": lambda out_fn: random_bitstring(out_fn, 256, 100000, 1000),
    "random-s-jaccard": lambda out_fn: random_jaccard(out_fn, n=10000, size=20, universe=40),
    "random-l-jaccard": lambda out_fn: random_jaccard(out_fn, n=100000, size=70, universe=100),
    "sift-128-euclidean": sift,
    "nytimes-256-angular": lambda out_fn: nytimes(out_fn, 256),
    "nytimes-16-angular": lambda out_fn: nytimes(out_fn, 16),
    "word2bits-800-hamming": lambda out_fn: word2bits(out_fn, "400K", "w2b_bitlevel1_size800_vocab400K"),
    "lastfm-64-dot": lambda out_fn: lastfm(out_fn, 64),
    "sift-256-hamming": lambda out_fn: sift_hamming(out_fn, "sift.hamming.256"),
    "kosarak-jaccard": lambda out_fn: kosarak(out_fn),
    "movielens1m-jaccard": movielens1m,
    "movielens10m-jaccard": movielens10m,
    "movielens20m-jaccard": movielens20m,
    "coco-i2i-512-angular": lambda out_fn: coco(out_fn, "i2i"),
    "coco-t2i-512-angular": lambda out_fn: coco(out_fn, "t2i"),
}

DATASETS.update({
    f"dbpedia-openai-{n//1000}k-angular": lambda out_fn, i=n: dbpedia_entities_openai_1M(out_fn, i)
    for n in range(100_000, 1_100_000, 100_000)
})
