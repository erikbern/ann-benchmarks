from __future__ import absolute_import

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

    def fit(self, X):
        pass

    def query(self, q, n):
        return [] # array of candidate indices
