import pytest
import numpy

from ann_benchmarks.distance import metrics


def test_euclidean():
    dist = metrics["euclidean"]["distance"]

    p = numpy.array([0, 1, 0])
    q = numpy.array([2, 0, 0])
    assert dist(p, q) == pytest.approx(5**0.5)


def test_hamming():
    dist = metrics["hamming"]["distance"]

    p = numpy.array([1, 1, 0, 0], dtype=numpy.bool_)
    q = numpy.array([1, 0, 0, 1], dtype=numpy.bool_)
    assert dist(p, q) == pytest.approx(2)

    p = numpy.array([1, 1, 0, 0])
    q = numpy.array([1, 0, 0, 1])
    assert dist(p, q) == pytest.approx(2)


def test_angular():
    dist = metrics["angular"]["distance"]

    p = numpy.array([5, 0])
    q = numpy.array([0, 3])
    assert dist(p, q) == pytest.approx(2**0.5)

    p = numpy.array([5, 0])
    q = numpy.array([1, 0])
    assert dist(p, q) == pytest.approx(0)

    p = numpy.array([7, 0])
    q = numpy.array([-8, 0])
    assert dist(p, q) == pytest.approx(2)
