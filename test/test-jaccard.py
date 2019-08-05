import unittest
import numpy
from ann_benchmarks.distance import jaccard, transform_dense_to_sparse

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

    def test_transformation(self):
        X = numpy.array([[True, False, False], [True, False, True], [False, False, True]])
        self.assertEqual(transform_dense_to_sparse(X), [[0],[0, 2], [2]])

