from __future__ import absolute_import
import sys
import importlib
import traceback

__constructors = [
  ("ann_benchmarks.algorithms.itu",
      ["ITUHashing", "ITUFilteringDouble"]),
  ("ann_benchmarks.algorithms.lshf",
      ["LSHF"]),
  ("ann_benchmarks.algorithms.annoy",
      ["Annoy"]),
  ("ann_benchmarks.algorithms.flann",
      ["FLANN"]),
  ("ann_benchmarks.algorithms.panns",
      ["PANNS"]),
  ("ann_benchmarks.algorithms.kdtree",
      ["KDTree"]),
  ("ann_benchmarks.algorithms.kgraph",
      ["KGraph"]),
  ("ann_benchmarks.algorithms.nearpy",
      ["NearPy"]),
  ("ann_benchmarks.algorithms.nmslib",
      ["NmslibNewIndex", "NmslibReuseIndex"]),
  ("ann_benchmarks.algorithms.falconn",
      ["FALCONN"]),
  ("ann_benchmarks.algorithms.balltree",
      ["BallTree"]),
  ("ann_benchmarks.algorithms.rpforest",
      ["RPForest"]),
  ("ann_benchmarks.algorithms.bruteforce",
      ["BruteForce", "BruteForceBLAS"]),
  ("ann_benchmarks.algorithms.subprocess",
      ["BitSubprocess", "BitSubprocessPrepared", "IntSubprocess", "FloatSubprocess"]),
  ("ann_benchmarks.algorithms.faiss",
      ["FaissLSH", "FaissIVF"]),
  ("ann_benchmarks.algorithms.faiss_gpu",
      ["FaissGPU"]),
  ("ann_benchmarks.algorithms.dolphinnpy",
      ["DolphinnPy"]),
  ("ann_benchmarks.algorithms.datasketch",
      ["DataSketch"])
]

available_constructors = {}
for name, symbols in __constructors:
    try:
        module = importlib.import_module(name)
        for symbol in symbols:
            assert hasattr(module, symbol), """\
import error: module %s does not define symbol %s""" % (name, symbol)
            available_constructors[symbol] = getattr(module, symbol)
    except ImportError:
        try:
            t, v, tb = sys.exc_info()
            traceback.print_exception(t, v, tb)
        finally:
            del tb
        print """\
warning: module %s could not be loaded, some algorithm constructors will not \
be available""" % name
        for symbol in symbols:
            available_constructors[symbol] = None
