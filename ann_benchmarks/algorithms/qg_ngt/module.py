import os
import subprocess
import time
import shutil
import psutil
import gc
import ctypes

import ngtpy

from ..base.module import BaseANN

class QG(BaseANN):
    def __init__(self, metric, object_type, epsilon, param):
        metrics = {"euclidean": "2", "angular": "E"}
        self._edge_size = param["edge"]
        self._outdegree = int(param["outdegree"])
        self._indegree = int(param["indegree"])
        self._outdegree_ext = int(param["outdegreeExt"]) if "outdegreeExt" in param.keys() else 0
        self._indegree_ext = int(param["indegreeExt"]) if "indegreeExt" in param.keys() else 0
        self._max_edge_size = int(param["max_edge"]) if "max_edge" in param.keys() else 128
        self._min_edge_size = int(param["min_edge"]) if "min_edge" in param.keys() else 64
        self._trim_edge_size = int(param["trim_edge"]) if "trim_edge" in param.keys() else 16
        self._metric = metrics[metric]
        self._object_type = object_type
        self._edge_size_for_search = int(param["search_edge"]) if "search_edge" in param.keys() else -2
        self._tree_disabled = (param["tree"] is False) if "tree" in param.keys() else False
        self._build_time_limit = float(param["timeout"]) if "timeout" in param.keys() else 4
        self._epsilon = float(param["epsilon"]) if "epsilon" in param.keys() else epsilon
        self._sample = int(param["sample"]) if "sample" in param.keys() else 20000
        self._leaf = param["leaf"] if "leaf" in param.keys() else '100:5'
        self._seed = param["seed"] if "seed" in param.keys() else '-10'
        self._reconst = param["reconst"] if "reconst" in param.keys() else 'base'
        print("QG: edge_size=" + str(self._edge_size))
        print("QG: outdegree=" + str(self._outdegree))
        print("QG: indegree=" + str(self._indegree))
        print("QG: edge_size_for_search=" + str(self._edge_size_for_search))
        print("QG: epsilon=" + str(self._epsilon))
        print("QG: metric=" + metric)
        print("QG: object_type=" + object_type)

    def fit(self, X):
        print("QG: start indexing...")
        dim = len(X[0])
        print("QG: # of data=" + str(len(X)))
        print("QG: dimensionality=" + str(dim))
        index_dir = "indexes"
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self._edge_size, self._outdegree, self._indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self._edge_size))
        forestIndex = os.path.join(index_dir, "FOREST-" + str(self._edge_size))
        index = forestIndex
        print("QG: index=" + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print("QG: create ANNG")
            t = time.time()
            args = [
                "ngt",
                "create",
                "-p8",
                "-b32",
                "-gs",
                "-oauto",
                "-D" + self._metric,
                "-d" + str(dim),
                "-E" + self._edge_size,
                "-S-1",
                "-e" + str(self._epsilon),
                "-P0",
                "-B30",
                "-rd",
                "-L" + self._leaf,
                "-s" + self._seed,
                anngIndex,
            ]
            print(" ".join(args))
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=24, build=False, debug=False)
            idx.save()
            idx.close()
            print("QG: insertion time(sec)=" + str(time.time() - t))
            del idx
            del X
            gc.collect()
            ngtpy.releaseMemory()
            t = time.time()
            args = [
                "ngt",
                "build-forest",
                anngIndex,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: construct Forest time(sec)=" + str(time.time() - t))
        if not os.path.exists(index):
            t = time.time()
            print("QG: degree adjustment")
            args = [
                "ngt",
                "reconstruct-graph",
                "-mS",
                "-E " + str(self._min_edge_size),
                "-o " + str(self._outdegree),
                "-i " + str(self._indegree),
                anngIndex,
                index,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: degree adjustment time(sec)=" + str(time.time() - t))
        try:
            del X
        except NameError:
            pass
        gc.collect()
        if not os.path.exists(index + "/qg"):
            print("QG: create and append...")
            t = time.time()
            args = [
                "qbg",
                "create-qg",
                "-R-:u",
                "-k0",
                index]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: create qg time(sec)=" + str(time.time() - t))
            print("QB: build...")
            t = time.time()
            args = [
                "qbg",
                "build-qg",
                "-o" + str(self._sample),
                "-M1",
                "-ib",
                "-I400",
                "-Gz",
                "-Pn",
                "-E" + str(self._max_edge_size),
                "-T" + str(self._trim_edge_size),
                index,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: build qg time(sec)=" + str(time.time() - t))
        if os.path.exists(index + "/qg/grp"):
            print("QG: index already exists! " + str(index))
            process = psutil.Process(os.getpid())
            vms = process.memory_info().vms
            rss = process.memory_info().rss
            print(f"QG: VMS(1): {vms / 1024**3:.2f} GiB")
            print(f"QG: RSS(1): {rss / 1024**3:.2f} GiB")
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print("QG: open time(sec)=" + str(time.time() - t))
            process = psutil.Process(os.getpid())
            vms = process.memory_info().vms
            rss = process.memory_info().rss
            print(f"QG: VMS(2): {vms / 1024**3:.2f} GiB")
            print(f"QG: RSS(2): {rss / 1024**3:.2f} GiB")
        else:
            print("QG: something wrong!?")
        print("QG: end of fit")

    def set_query_arguments(self, parameters):
        shift = 0
        scale = 1
        if len(parameters) == 2:
            result_expansion, epsilon = parameters
        elif len(parameters) == 3:
            result_expansion, epsilon, shift = parameters
        else:
            result_expansion, epsilon, shift, scale = parameters
        self.name = "QG-NGT(%s,%s,%s:%s,%s:%s,%s,%s,%s,%s,%s,%s,%s,%s)" % (
            self._edge_size,
            self._epsilon,
            self._outdegree,
            self._indegree,
            self._outdegree_ext,
            self._indegree_ext,
            self._max_edge_size,
            self._sample,
            self._leaf,
            self._seed,
            result_expansion,
            epsilon,
            shift,
            scale,
        )
        epsilon = epsilon - 1.0
        print("QG: result_expansion=" + str(result_expansion))
        print("QG: epsilon=" + str(epsilon))
        print("QG: shift=" + str(shift))
        print("QG: scale=" + str(scale))
        self.index.set(epsilon=epsilon, result_expansion=result_expansion, id_bit_shift=shift,
                       distance_lut_scale_factor=scale)

    def query(self, v, n):
        return self.index.search(v, n)

    def freeIndex(self):
        print("QG: free")
