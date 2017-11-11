import h5py
import numpy
import os
import random
import struct
import sys
import tarfile
import urllib.request
import zipfile

sys.path.append('.')
from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        urllib.request.urlretrieve(src, dst)


def write_output(train, test, fn, distance, count=100):
    n = 0
    f = h5py.File(fn, 'w')
    f.create_dataset('train', (len(train), len(train[0])), dtype='f')[:] = train
    f.create_dataset('test', (len(test), len(test[0])), dtype='f')[:] = test
    correct = f.create_dataset('correct', (len(test), count), dtype='i')
    bf = BruteForceBLAS(distance, precision=numpy.float32)
    bf.fit(train)
    queries = []
    for i, x in enumerate(test):
        if i % 1000 == 0:
            print('%d...' % i)
        correct[i] = [j for j, _ in bf.query_with_distances(x, count)]
    f.close()


def glove(tag, d):
    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        random.seed(1)
        print('parsing %s' % tag)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        train = []
        test = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            if random.randint(0, 4):
                train.append(v)
            else:
                test.append(v)

        print('writing output...')
        train = numpy.array(train)
        test = numpy.array(test)
        write_output(train, test, '%s.hdf5' % tag, 'angular')


def sift(tag, ):
    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        def load_vectors(t_fn):
            print('reading %s...' % t_fn)
            vs = []
            f = t.extractfile(t.getmember(t_fn))
            while True:
                b = f.read(4)
                if not b:
                    break
                dim = struct.unpack('i', b)[0]
                vec = struct.unpack('f' * dim, f.read(dim*4))
                vs.append(vec)
            return numpy.array(vs)
            
        train = load_vectors('sift/sift_base.fvecs')
        test = load_vectors('sift/sift_query.fvecs')
        write_output(train, test, '%s.hdf5' % tag, 'euclidean')


datasets = {
    'sift-128-euclidean': sift,
    'glove-25-angular': lambda tag: glove(tag, 25),
    'glove-50-angular': lambda tag: glove(tag, 50),
    'glove-100-angular': lambda tag: glove(tag, 100),
    'glove-200-angular': lambda tag: glove(tag, 200),
}

if __name__ == '__main__':
    tag = sys.argv[1]
    if tag not in datasets:
        raise Exception('%s has to be one of %s' % (tag, ','.join(datasets.keys())))
    datasets[tag](tag)
