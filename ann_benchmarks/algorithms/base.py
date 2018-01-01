from __future__ import absolute_import
import psutil

class BaseANN(object):
    def use_threads(self):
        return True

    def done(self):
        pass

    def batch_query(self, X, n):
        res = []
        for q in X:
            res.append(self.query(q, n))
        return res

    def get_index_size(self, process):
        """Returns the size of the index in kB or -1 if not implemented."""
        return psutil.Process().memory_info().rss / 1024  # return in kB for backwards compatibility

    def fit(self, X):
        pass

    def query(self, q, n):
        return [] # array of candidate indices
