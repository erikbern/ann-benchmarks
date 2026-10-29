import diskannpy
import tempfile

from ..base.module import BaseANN


class DiskANN(BaseANN):
    def __init__(self, metric, method_param):
        self.name = None
        self.search_complexity = None
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        self.index = None
        self.tempdir = tempfile.TemporaryDirectory()

    def fit(self, X):
        max_degree = self.method_param.get("max_degree", 64)
        complexity = self.method_param.get("complexity", 128)
        alpha = self.method_param.get("alpha", 1.2)

        diskannpy.build_memory_index(
            data=X,
            distance_metric=self.metric,
            index_directory=self.tempdir.name,
            complexity=complexity, # candidate NN list when building. Typically 75-200. At least as large as graph degree.
            graph_degree=max_degree, # max graph degree. Typically 60-150. Higher means better recall.
            num_threads=0, # uses max available processors
            alpha=alpha, # controls number of points added to the graph
            use_pq_build=False, # uses quantization to save index which reduces recall and disk space
            use_opq=False,
        )

        self.index = diskannpy.StaticMemoryIndex(
            index_directory=self.tempdir.name,
            num_threads=1,
            initial_search_complexity=100, # most common complexity durig search. Working mem is initialized based off this * threads.
            distance_metric=self.metric,
            dimensions = None
        )

    def set_query_arguments(self, complexity):
        self.search_complexity = complexity
        self.name = "diskann (%s, 'complexity': %s)" % (self.method_param, complexity)

    def query(self, v, n):
        ids, distances = self.index.search(v, n, self.search_complexity)
        return ids


    def done(self):
        self.tempdir.cleanup()
        pass
