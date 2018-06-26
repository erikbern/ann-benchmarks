from __future__ import absolute_import
import psutil

class BaseANN(object):
    def done(self):
        pass

    def get_index_size(self, process):
        """Returns the size of the index in kB or -1 if not implemented."""
        return psutil.Process().memory_info().rss / 1024  # return in kB for backwards compatibility

    def fit(self, X):
        pass

    def query(self, q, n):
        return [] # array of candidate indices

    def batch_query(self, X, n):
        self.res = []
        for q in X:
            self.res.append(self.query(q, n))

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return self.name
