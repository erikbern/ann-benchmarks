from __future__ import absolute_import
from ctypes import *
from ann_benchmarks.algorithms.base import BaseANN

UCX_ANN_SO_PATH = "ucx/build/lib/libucs_ann_search.so"
UCX_ANN_FUNC_FORMAT = "{prefix}nn_{cmd}_{key}_to_{val}_{dim}_dim_k{k}"
KHASH_INIT_STRING = "%i-dimensional Khash on %s (%s metric, k=%i)"

TYPE_MAP = {
    "int"      : c_int,
    "unsigned" : c_uint,
    "char"     : c_byte,
    "ptr"      : c_longlong,
    "int64"    : c_longlong,
    "cstr"     : c_char_p,
    "float"    : c_float,
    "bit"      : c_byte
}

class KhashDB(object):
    def __invoke(self, cmd, ret_type, *args):
        func_name = UCX_ANN_FUNC_FORMAT.format(prefix=self._prefix,
                                               cmd=cmd,
                                               key=self._key,
                                               val=self._value,
                                               dim=self._dim,
                                               k=self._k)
        func = self._so.__getattr__(func_name)
        func.restype = ret_type
        ret = func(*args)
        if ret_type is not None:
            ret = ret_type(ret)
        return ret

    def __init__(self, so, k, dim, point_type, nn_type,
                 value="unsigned", coefficient="float"):
        self._so          = so
        self._k           = k
        self._dim         = dim
        if point_type == "bit":
            point_type    = "char"
        self._key         = point_type
        self._key_type    = TYPE_MAP[point_type] * dim
        self._value       = value
        self._value_type  = TYPE_MAP[value]
        self._search_type = TYPE_MAP[value] * k
        self._coefficient = coefficient
        self._size        = 0
        self._prefix      = nn_type

        if nn_type == "w":
            c_arg        = TYPE_MAP[coefficient](1)
            c_array_type = TYPE_MAP[coefficient] * dim
            c_array_arg  = [1]*dim
            init_args    = [c_arg, byref(c_array_type(*c_array_arg))]
        elif nn_type == "a":
            c_arg        = TYPE_MAP[coefficient](1)
            init_args    = [c_arg]
        elif nn_type == "k":
            init_args    = []
        else:
            raise ValueError("\"%s\" given, but nn_type can only be w/a/k")

        self._db = self.__invoke("init", c_void_p, *init_args)

    def __del__(self):
        self.__invoke("destroy", None, self._db)

    def insert(self, key):
        ret = self.__invoke("insert", c_int, self._db,
                            byref(self._key_type(*key)),
                            self._value_type(self._size))
        assert(ret.value == 0)
        self._size += 1
        if ((self._size % 1000) == 0):
            print(self._size)

    def resize(self, length):
        self.__invoke("resize", c_int, self._db, c_uint(length))

    def search(self, key):
        key = [int(x) for x in key]
        key = byref(self._key_type(*key))
        knn = self._search_type()
        self.__invoke("search", c_int, self._db, key, byref(knn))
        return knn

class Khash(BaseANN):
    def __init__(self, k, metric, dimension, point_type, nn_type=None):
        self.name = KHASH_INIT_STRING % (dimension, point_type, metric, k)

        # Choose between KNN, ANN (C-approximate KNN) and WNN (Weighted ANN)
        if nn_type is None:
            if metric == "weighted-hamming": # not yet supported in ann-benchmark
                nn_type = "w"
            else:
                nn_type = "k"

        self._db = KhashDB(cdll.LoadLibrary(UCX_ANN_SO_PATH), k, dimension,
                           point_type, nn_type)

    def fit(self, X):
        self._db.resize(len(X))
        for v in X:
            self._db.insert(v)

    def query(self, v, n):
        return self._db.search(v)
