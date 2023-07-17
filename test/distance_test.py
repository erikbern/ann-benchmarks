import pytest
import numpy

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.distance import metrics


def test_euclidean():
    dist = metrics["euclidean"].distance

    p = numpy.array([0, 1, 0])
    q = numpy.array([2, 0, 0])
    assert dist(p, q) == pytest.approx(5**0.5)


def test_hamming():
    dist = metrics["hamming"].distance

    p = numpy.array([1, 1, 0, 0], dtype=numpy.bool_)
    q = numpy.array([1, 0, 0, 1], dtype=numpy.bool_)
    assert dist(p, q) == pytest.approx(2)

    p = numpy.array([1, 1, 0, 0])
    q = numpy.array([1, 0, 0, 1])
    assert dist(p, q) == pytest.approx(2)


def test_angular():
    dist = metrics["angular"].distance

    # We use 1 - cos as the angular distance

    p = numpy.array([5, 0])
    q = numpy.array([0, 3])
    assert dist(p, q) == pytest.approx(1)

    p = numpy.array([5, 0])
    q = numpy.array([1, 0])
    assert dist(p, q) == pytest.approx(0)

    p = numpy.array([7, 0])
    q = numpy.array([-8, 0])
    assert dist(p, q) == pytest.approx(2)


def test_angular_dataset():
    # Make sure distances in the datasets are calculated consistent with the definitions
    # This is to avoid issues like #367

    dist_f = metrics["angular"].distance

    hdf5_f, n_dims = get_dataset("glove-25-angular")

    n = 1000
    for u, nns, dists in zip(hdf5_f["test"][:n], hdf5_f["neighbors"][:n], hdf5_f["distances"][:n]):
        for j, dist in zip(nns, dists):
            v = hdf5_f["train"][j]
            assert dist_f(u, v) == pytest.approx(dist, rel=1e-4, abs=1e-4)
