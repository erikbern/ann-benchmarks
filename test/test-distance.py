import unittest
import numpy

from ann_benchmarks.distance import euclidean

class TestDistance(unittest.TestCase):
    def test_euclidean(self):
        p = numpy.array([0, 1, 0])
        q = numpy.array([2, 0, 0])
        self.assertAlmostEqual(euclidean(p, q), 5 ** 0.5)
