import pytest
import numpy

from ann_benchmarks.distance import metrics


def test_euclidean():
    p = numpy.array([0, 1, 0])
    q = numpy.array([2, 0, 0])
    dist = metrics["euclidean"]["distance"]
    assert dist(p, q) == pytest.approx(5**0.5)


def test_hamming():
    p = numpy.array([1, 1, 0, 0], dtype=numpy.bool_)
    q = numpy.array([1, 0, 0, 1], dtype=numpy.bool_)
    dist = metrics["hamming"]["distance"]
    assert dist(p, q) == pytest.approx(2)

    p = numpy.array([1, 1, 0, 0])
    q = numpy.array([1, 0, 0, 1])
    assert dist(p, q) == pytest.approx(2)
