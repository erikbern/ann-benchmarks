from __future__ import absolute_import
import os
import gzip
import yaml
import numpy

from ann_benchmarks.data import type_info

def get_dataset(which='glove', limit=-1):
    cache = 'queries/%s-%d.npz' % (which, limit)
    try:
        with numpy.load(cache) as f:
            X = f["data"]
            manifest = f["manifest"][0]
            return manifest, X
    except IOError:
        pass
    except KeyError:
        pass

    local_fn = os.path.join('install', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    manifest = get_manifest(which)
    point_type = manifest['dataset']['point_type']

    assert point_type in type_info, """\
dataset %s: unknown point type '%s'""" % (which, point_type)
    info = type_info[point_type]

    assert "parse_entry" in info, """\
dataset %s: no parser for points of type '%s'""" % (which, point_type)
    loader = info["parse_entry"]

    X = []
    for i, line in enumerate(f):
        X.append(loader(line))
        if limit != -1 and len(X) == limit:
            break
    X = numpy.array(X, dtype=info.get("type"))

    if "finish_entries" in info:
        X = info["finish_entries"](X)

    numpy.savez(cache, manifest=[manifest], data=X)
    return manifest, X

def get_manifest(which):
    local_fn = os.path.join('install', which)
    manifest = {
        'dataset': {
            'point_type': 'float',
            'test_size' : 10000
        }
    }
    if os.path.exists(local_fn + '.yaml'):
        with open(local_fn + '.yaml') as mf:
            r = yaml.load(mf)
            for name, value in r.items():
                if name in manifest:
                    manifest[name].update(value)
                else:
                    manifest[name] = value
    return manifest

def split_dataset(X, random_state=3, test_size=10000):
    import sklearn.cross_validation

    # Here Erik is most welcome to use any other random_state
    # However, it is best to use a new random seed for each major re-evaluation,
    # so that we test on a trully bind data.
    X_train, X_test = \
      sklearn.cross_validation.train_test_split(
          X, test_size=test_size, random_state=random_state)
    print(X_train.shape, X_test.shape)

    return X_train, X_test

def get_query_cache_path(dataset, count, limit, distance, query_dataset = None):
    if not query_dataset:
        return "queries/%s_%s_%s_%s.p" % (dataset, count, limit, distance)
    else:
        return \
          "queries/%s_%s_%s_%s_%s.p" % (dataset, count, limit, query_dataset, distance)
