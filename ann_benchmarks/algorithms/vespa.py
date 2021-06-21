from ann_benchmarks.algorithms.base import BaseANN
from vespa_ann_benchmark import DistanceMetric, HnswIndexParams, HnswIndex
import time

# Class using the Vespa implementation of an HNSW index for nearest neighbor
# search over data points in a high dimensional vector space.
#
# To use nearest neighbor search in a Vespa application,
# see https://docs.vespa.ai/en/approximate-nn-hnsw.html for more details.
class VespaHnswBase(BaseANN):
    def __init__(self, enable_normalize, metric, dimension, param):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError(
                "VespaHnsw doesn't support metric %s" % metric)
        self.metric = {'angular': DistanceMetric.Angular, 'euclidean': DistanceMetric.Euclidean}[metric]
        normalize = False
        if self.metric == DistanceMetric.Angular and enable_normalize:
            normalize = True
            self.metric = DistanceMetric.InnerProduct
        self.param = param
        self.neighbors_to_explore_at_insert = param.get("efConstruction", 200)
        self.max_links_per_node = param.get("M", 8)
        self.dimension = dimension
        self.neighbors_to_explore = 200
        self.name = 'VespaHnsw()'
        self.index = HnswIndex(dimension, HnswIndexParams(self.max_links_per_node, self.neighbors_to_explore_at_insert, self.metric, False), normalize)

    def fit(self, X):
        for i, x in enumerate(X):
            self.index.set_vector(i, x)

    def set_query_arguments(self, ef):
        print("VespaHnsw: ef = " + str(ef))
        self.neighbors_to_explore = ef

    def query(self, v, n):
        return [index for index, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        return self.index.find_top_k(n, v, self.neighbors_to_explore)

class VespaHnsw(VespaHnswBase):
    def __init__(self, metric, dimension, param):
        super().__init__(True, metric, dimension, param)

    def __str__(self):
        return 'VespaHnsw ({}, ef: {})'.format(self.param, self.neighbors_to_explore)
