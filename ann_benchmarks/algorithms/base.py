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

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance (in kilobytes), or None if this information is not available."""
        return psutil.Process().memory_info().rss / 1024  # return in kB for backwards compatibility

    def fit(self, X):
        pass

    def query(self, q, n):
        return [] # array of candidate indices

    def __str__(self):
        return self.name
