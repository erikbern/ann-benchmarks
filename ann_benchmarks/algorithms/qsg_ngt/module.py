import os
import struct
import subprocess
import time

import ngtpy
from sklearn import preprocessing

from ..base.module import BaseANN


class QSG(BaseANN):
    def __init__(self, metric, object_type, epsilon, param):
        metrics = {"euclidean": "2", "angular": "E"}
        self._edge_size = int(param["edge"])
        self._outdegree = int(param["outdegree"])
        self._indegree = int(param["indegree"])
        self._max_edge_size = int(param["max_edge"]) if "max_edge" in param.keys() else 128
        self._metric = metrics[metric]
        self._object_type = object_type
        self._edge_size_for_search = int(param["search_edge"]) if "search_edge" in param.keys() else -2
        self._tree_disabled = (param["tree"] is False) if "tree" in param.keys() else False
        self._build_time_limit = float(param["timeout"]) if "timeout" in param.keys() else 4
        self._epsilon = float(param["epsilon"]) if "epsilon" in param.keys() else epsilon
        self._sample = int(param["sample"]) if "sample" in param.keys() else 20000
        self._paramE = param["paramE"]
        self._paramS = param["paramS"]
        self._range = int(param["range"])
        self._threshold = int(param["threshold"])
        self._rangeMax = int(param["rangeMax"])
        self._searchA = int(param["searchA"])
        self._ifES = int(param["ifES"])
        print("QSG: edge_size=" + str(self._edge_size))
        print("QSG: outdegree=" + str(self._outdegree))
        print("QSG: indegree=" + str(self._indegree))
        print("QSG: edge_size_for_search=" + str(self._edge_size_for_search))
        print("QSG: epsilon=" + str(self._epsilon))
        print("QSG: metric=" + metric)
        print("QSG: object_type=" + object_type)
        print("QSG: range=" + str(self._range))
        print("QSG: threshold=" + str(self._threshold))

    def fit(self, X):
        print("QSG: start indexing...")
        dim = len(X[0])
        print("QSG: # of data=" + str(len(X)))
        print("QSG: dimensionality=" + str(dim))
        index_dir = "indexes"
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self._edge_size, self._outdegree, self._indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self._edge_size))
        print("QSG: index=" + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print("QSG: create ANNG")
            t = time.time()
            args = [
                "ngt",
                "create",
                "-it",
                "-p8",
                "-b500",
                "-ga",
                "-of",
                "-D" + self._metric,
                "-d" + str(dim),
                "-E" + str(self._edge_size),
                "-S40",
                "-e" + str(self._epsilon),
                "-P0",
                "-B30",
                "-T" + str(self._build_time_limit),
                "-R" + str(self._range),
                "-t" + str(self._threshold),
                "-M" + str(self._rangeMax),
                "-A" + str(self._searchA),
                "-H" + str(self._ifES),
                anngIndex,
            ]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=24, debug=False)
            idx.save()
            idx.close()
            print("QSG: ANNG construction time(sec)=" + str(time.time() - t))
        if self._ifES == 1:
            if self._metric == "E":
                X_normalized = preprocessing.normalize(X, norm="l2")
                fvecs_dir = "fvecs"
                if not os.path.exists(fvecs_dir):
                    os.makedirs(fvecs_dir)
                fvecs = os.path.join(fvecs_dir, "base.fvecs")
                with open(fvecs, "wb") as fp:
                    for y in X_normalized:
                        d = struct.pack("I", y.size)
                        fp.write(d)
                        for x in y:
                            a = struct.pack("f", x)
                            fp.write(a)
            else:
                fvecs_dir = "fvecs"
                if not os.path.exists(fvecs_dir):
                    os.makedirs(fvecs_dir)
                fvecs = os.path.join(fvecs_dir, "base.fvecs")
                with open(fvecs, "wb") as fp:
                    for y in X:
                        d = struct.pack("I", y.size)
                        fp.write(d)
                        for x in y:
                            a = struct.pack("f", x)
                            fp.write(a)
            parmEfanna = self._paramE
            parmSSG = self._paramS
            graph_dir = "graph"
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            KNNG = os.path.join(
                graph_dir,
                "KNNG-"
                + str(parmEfanna[0])
                + "-"
                + str(parmEfanna[1])
                + "-"
                + str(parmEfanna[2])
                + "-"
                + str(parmEfanna[3])
                + "-"
                + str(parmEfanna[4])
                + ".graph",
            )
            SG = os.path.join(anngIndex, "grp")
            cmds = (
                "/home/app/HWTL_SDU-ANNS/qsgngt-knng "
                + str(fvecs)
                + " "
                + str(KNNG)
                + " "
                + str(parmEfanna[0])
                + " "
                + str(parmEfanna[1])
                + " "
                + str(parmEfanna[2])
                + " "
                + str(parmEfanna[3])
                + " "
                + str(parmEfanna[4])
                + "&& /home/app/HWTL_SDU-ANNS/qsgngt-SpaceGraph "
                + str(fvecs)
                + " "
                + str(KNNG)
                + " "
                + str(parmSSG[0])
                + " "
                + str(parmSSG[1])
                + " "
                + str(parmSSG[2])
                + " "
                + str(SG)
            )
            os.system(cmds)

        if not os.path.exists(index):
            print("QSG: degree adjustment")
            t = time.time()
            args = [
                "ngt",
                "reconstruct-graph",
                "-mS",
                "-E " + str(self._outdegree),
                "-o " + str(self._outdegree),
                "-i " + str(self._indegree),
                anngIndex,
                index,
            ]
            subprocess.call(args)
            print("QSG: degree adjustment time(sec)=" + str(time.time() - t))
        if not os.path.exists(index + "/qg"):
            print("QSG:create and append...")
            t = time.time()
            args = ["qbg", "create-qg", index]
            subprocess.call(args)
            print("QSG: create qg time(sec)=" + str(time.time() - t))
            print("QB: build...")
            t = time.time()
            args = [
                "qbg",
                "build-qg",
                "-o" + str(self._sample),
                "-M6",
                "-ib",
                "-I400",
                "-Gz",
                "-Pn",
                "-E" + str(self._max_edge_size),
                index,
            ]
            subprocess.call(args)
            print("QSG: build qg time(sec)=" + str(time.time() - t))
        if os.path.exists(index + "/qg/grp"):
            print("QSG: index already exists! " + str(index))
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print("QSG: open time(sec)=" + str(time.time() - t))
        else:
            print("QSG: something wrong.")
        print("QSG: end of fit")
        print("QSG:Successfully Build Index")

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        print("QSG: result_expansion=" + str(result_expansion))
        print("QSG: epsilon=" + str(epsilon))
        self.name = "QSG-NGT(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %1.3f)" % (
            self._edge_size,
            self._outdegree,
            self._indegree,
            self._max_edge_size,
            str(self._range),
            str(self._threshold),
            str(self._rangeMax),
            str(self._searchA),
            str(self._ifES),
            epsilon,
            result_expansion,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def freeIndex(self):
        print("QSG: free")
