from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
import struct
import numpy as np
import click
import h5py
from joblib import Parallel, delayed
import multiprocessing

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
 
def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
 
def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)
 
        
def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)

def calc_i(i, x, bf, test, neighbors, distances, count):
    if i % 1000 == 0:
        print('%d/%d...' % (i, len(test)))
    res = list(bf.query_with_distances(x, count))
    res.sort(key=lambda t: t[-1])
    neighbors[i] = [j for j, _ in res]
    distances[i] = [d for _, d in res]


def calc(bf, test, neighbors, distances, count):
    Parallel(n_jobs=multiprocessing.cpu_count(),  require='sharedmem')(delayed(calc_i)(i, x, bf, test, neighbors, distances, count) for i, x in enumerate(test))


def write_output(train, test, fn, distance, point_type='float', count=100):
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
    calc(bf, test, neighbors, distances, count)
    f.close()

@click.command()
@click.option('--size', default=10, help='Number of vectors in milions.')
@click.option('--distance', default='angular', help='distance metric.')
@click.option('--test_set', required=True, type=str)
@click.option('--train_set', required=True, type=str)
def create_ds(size, distance, test_set, train_set):
    test_set = read_fbin(test_set)
    train_set= read_fbin(train_set, chunk_size=size*1000000)
    write_output(train=train_set, test=test_set, fn=f'Text-to-Image-{size}M.hd5f', distance=distance, point_type='float', count=100)

if __name__ == "__main__":
    create_ds()