from __future__ import absolute_import
from scipy.spatial.distance import pdist as scipy_pdist
import numpy

def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]

def jaccard(a, b):
    return len(numpy.intersect1d(a, b)) / float(len(numpy.union1d(a,b)))


metrics = {
    'hamming': lambda a, b: pdist(a, b, "hamming"),
    'jaccard': lambda a, b: jaccard(a, b),
    'euclidean': lambda a, b: pdist(a, b, "euclidean"),
    'angular': lambda a, b: pdist(a, b, "cosine")
}
