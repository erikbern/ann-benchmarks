import unittest
import numpy

from ann_benchmarks.distance import metrics

class TestDistance(unittest.TestCase):
    def test_euclidean(self):
        p = numpy.array([0, 1, 0])
        q = numpy.array([2, 0, 0])
        dist = metrics["euclidean"]["distance"]
        self.assertAlmostEqual(dist(p, q), 5 ** 0.5)

    def test_hamming(self):
        p = numpy.array([1, 1, 0, 0], dtype=numpy.bool_)
        q = numpy.array([1, 0, 0, 1], dtype=numpy.bool_)
        dist = metrics["hamming"]["distance"]
        self.assertAlmostEqual(dist(p, q), 2)

        p = numpy.array([1, 1, 0, 0])
        q = numpy.array([1, 0, 0, 1])
        self.assertAlmostEqual(dist(p, q), 2)
