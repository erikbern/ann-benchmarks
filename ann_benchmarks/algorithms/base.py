from __future__ import absolute_import
import psutil


class BaseANN(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X):
        pass

    def query(self, q, n):
        return []  # array of candidate indices

    def batch_query(self, X, n):
        self.res = []
        for q in X:
            self.res.append(self.query(q, n))

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name
