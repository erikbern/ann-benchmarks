from __future__ import absolute_import
from os import sep as pathsep
import sys
import yaml
import traceback
from itertools import product
from ann_benchmarks.algorithms.constructors import available_constructors


def _handle_args(args):
    if isinstance(args, list):
        args = map(lambda el: el if isinstance(el, list) else [el], args)
        return map(list, product(*args))
    elif isinstance(args, dict):
        flat = []
        for k, v in args.items():
            if isinstance(v, list):
                flat.append(map(lambda el: (k, el), v))
            else:
                flat.append([(k, v)])
        return map(dict, product(*flat))
    else:
        raise TypeError("No args handling exists for %s" % type(args).__name__)


def _get_definitions(definition_file):
    with open(definition_file, "r") as f:
        return yaml.load(f)


def list_algorithms(definition_file):
    definitions = _get_definitions(definition_file)

    print('The following algorithms are supported...')
    for point in definitions:
        print('\t... for the point type "%s"...' % point)
        for metric in definitions[point]:
            print('\t\t... and the distance metric "%s":' % metric)
            for algorithm in definitions[point][metric]:
                print('\t\t\t%s' % algorithm)


def get_algorithms(definition_file, dimension, point_type="float", distance_metric="euclidean", count=10):
    definitions = _get_definitions(definition_file)

    algorithm_definitions = {}
    if "any" in definitions[point_type]:
        algorithm_definitions.update(definitions[point_type]["any"])
    algorithm_definitions.update(definitions[point_type][distance_metric])

    algos = {}
    for (name, algo) in algorithm_definitions.items():
        assert "constructor" in algo, """\
group %s does not specify a constructor""" % name
        cn = algo["constructor"]
        assert cn in available_constructors, """\
group %s specifies the unknown constructor %s""" % (name, cn)
        constructor = available_constructors[cn]
        if not constructor:
            print('warning: group %s specifies the known, but missing, constructor %s; skipping' % (name, cn))
            continue

        algos[name] = []

        base_args = []
        if "base-args" in algo:
            base_args = algo["base-args"]

        for run_group in algo["run-groups"].values():
            if "arg-groups" in run_group:
                groups = []
                for arg_group in run_group["arg-groups"]:
                    if isinstance(arg_group, dict):
                        # Dictionaries need to be expanded into lists in order
                        # for the subsequent call to _handle_args to do the
                        # right thing
                        groups.append(_handle_args(arg_group))
                    else:
                        groups.append(arg_group)
                args = _handle_args(groups)
            elif "args" in run_group:
                args = _handle_args(run_group["args"])
            else:
                assert False, "? what? %s" % run_group

            for arg_group in args:
                obj = None
                try:
                    aargs = []
                    aargs.extend(base_args)
                    if isinstance(arg_group, list):
                        aargs.extend(arg_group)
                    else:
                        aargs.append(arg_group)

                    vs = {
                        "@count": count,
                        "@metric": distance_metric,
                        "@dimension": dimension
                    }
                    def _handle(arg):
                        if isinstance(arg, dict):
                            return dict([(k, _handle(v)) for k, v in arg.items()])
                        elif isinstance(arg, list):
                            return map(_handle, arg)
                        elif isinstance(arg, str) and arg in vs:
                            return vs[arg]
                        else:
                            return arg
                    aargs = map(_handle, aargs)
                    obj = constructor(*aargs)
                    if not obj.name:
                        raise Exception("""\
algorithm instance "%s" does not have a name""" % obj)
                    elif pathsep in obj.name:
                        raise Exception("""\
algorithm instance "%s" has an invalid name (it contains a path \
separator)""" % obj.name)
                    algos[name].append(obj)
                except Exception:
                    try:
                        t, v, tb = sys.exc_info()
                        traceback.print_exception(t, v, tb)
                    finally:
                        del tb
                    print('warning: constructor %s (with parameters %s) failed, skipping' % (cn, str(aargs)))
    return algos
