import unittest
import numpy
from ann_benchmarks.distance import jaccard

class TestJaccard(unittest.TestCase):
    def setUp(self):
        pass

    def test_similarity(self):
        a = [1, 2, 3, 4]
        b = []
        c = [1, 2]
        d = [5, 6]

        self.assertAlmostEqual(jaccard(a, b), 0.0)
        self.assertAlmostEqual(jaccard(a, a), 1.0)
        self.assertAlmostEqual(jaccard(a, c), 0.5)
        self.assertAlmostEqual(jaccard(c, d), 0.0)

