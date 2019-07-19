from __future__ import absolute_import
from scipy.spatial.distance import pdist as scipy_pdist
import numpy as np

def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]

# Need own implementation of jaccard because scipy's
# implementation is different

def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
         return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)

def transform_dense_to_sparse(X):
    """Converts the n * m dataset into a sparse format
    that only holds the non-zero entries (Jaccard)."""
    # get list of indices of non-zero elements
    indices = np.transpose(np.where(X))
    keys = []
    l = []
    last_i = None
    for i, j in indices:
        if last_i != None and last_i != i:
            keys.append(l)
            l = []
        l.append(j)
        last_i = i
    keys.append(l)

    assert len(X) == len(keys)

    return keys

metrics = {
    'hamming': {
        'distance': lambda a, b: pdist(a, b, "hamming"),
        'distance_valid': lambda a: True
    },
    # return 1 - jaccard similarity, because smaller distances are better.
    'jaccard': {
        'distance': lambda a, b: 1 - jaccard(a, b),
        'distance_valid': lambda a: a < 1 - 1e-5
    },
    'euclidean': {
        'distance': lambda a, b: pdist(a, b, "euclidean"),
        'distance_valid': lambda a: True
    },
    'angular': {
        'distance': lambda a, b: pdist(a, b, "cosine"),
        'distance_valid': lambda a: True
    }
}

dataset_transform = {
    'hamming': lambda X: X,
    'euclidean': lambda X: X,
    'angular': lambda X: X,
    'jaccard' : lambda X: transform_dense_to_sparse(X)
}
