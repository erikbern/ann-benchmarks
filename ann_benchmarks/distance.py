from __future__ import absolute_import
from scipy.spatial.distance import pdist as scipy_pdist


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]

# Need own implementation of jaccard because numpy's
# implementation is different


def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(a & b)
    return intersect / (float)(len(a) + len(b) - intersect)


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
