from __future__ import absolute_import
from os import sep as pathsep
import collections
import importlib
import os
import sys
import traceback
import yaml
from enum import Enum
from itertools import product


Definition = collections.namedtuple(
    'Definition',
    ['algorithm', 'constructor', 'module', 'docker_tag',
     'arguments', 'query_argument_groups', 'disabled'])


def get_algorithm_name(name, batch):
    if batch:
        return name + "-batch"
    return name


def instantiate_algorithm(definition):
    print('Trying to instantiate %s.%s(%s)' %
          (definition.module, definition.constructor, definition.arguments))
    module = importlib.import_module(definition.module)
    constructor = getattr(module, definition.constructor)
    return constructor(*definition.arguments)


class InstantiationStatus(Enum):
    AVAILABLE = 0
    NO_CONSTRUCTOR = 1
    NO_MODULE = 2


def algorithm_status(definition):
    try:
        module = importlib.import_module(definition.module)
        if hasattr(module, definition.constructor):
            return InstantiationStatus.AVAILABLE
        else:
            return InstantiationStatus.NO_CONSTRUCTOR
    except ImportError:
        return InstantiationStatus.NO_MODULE


def _generate_combinations(args):
    if isinstance(args, list):
        args = [el if isinstance(el, list) else [el] for el in args]
        return [list(x) for x in product(*args)]
    elif isinstance(args, dict):
        flat = []
        for k, v in args.items():
            if isinstance(v, list):
                flat.append([(k, el) for el in v])
            else:
                flat.append([(k, v)])
        return [dict(x) for x in product(*flat)]
    else:
        raise TypeError("No args handling exists for %s" % type(args).__name__)


def _substitute_variables(arg, vs):
    if isinstance(arg, dict):
        return dict([(k, _substitute_variables(v, vs))
                     for k, v in arg.items()])
    elif isinstance(arg, list):
        return [_substitute_variables(a, vs) for a in arg]
    elif isinstance(arg, str) and arg in vs:
        return vs[arg]
    else:
        return arg


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


def get_unique_algorithms(definition_file):
    definitions = _get_definitions(definition_file)
    algos = set()
    for point in definitions:
        for metric in definitions[point]:
            for algorithm in definitions[point][metric]:
                algos.add(algorithm)
    return list(sorted(algos))


def get_definitions(definition_file, dimension, point_type="float",
                    distance_metric="euclidean", count=10):
    definitions = _get_definitions(definition_file)

    algorithm_definitions = {}
    if "any" in definitions[point_type]:
        algorithm_definitions.update(definitions[point_type]["any"])
    algorithm_definitions.update(definitions[point_type][distance_metric])

    definitions = []
    for (name, algo) in algorithm_definitions.items():
        for k in ['docker-tag', 'module', 'constructor']:
            if k not in algo:
                raise Exception(
                    'algorithm %s does not define a "%s" property' % (name, k))

        base_args = []
        if "base-args" in algo:
            base_args = algo["base-args"]

        for run_group in algo["run-groups"].values():
            if "arg-groups" in run_group:
                groups = []
                for arg_group in run_group["arg-groups"]:
                    if isinstance(arg_group, dict):
                        # Dictionaries need to be expanded into lists in order
                        # for the subsequent call to _generate_combinations to
                        # do the right thing
                        groups.append(_generate_combinations(arg_group))
                    else:
                        groups.append(arg_group)
                args = _generate_combinations(groups)
            elif "args" in run_group:
                args = _generate_combinations(run_group["args"])
            else:
                assert False, "? what? %s" % run_group

            if "query-arg-groups" in run_group:
                groups = []
                for arg_group in run_group["query-arg-groups"]:
                    if isinstance(arg_group, dict):
                        groups.append(_generate_combinations(arg_group))
                    else:
                        groups.append(arg_group)
                query_args = _generate_combinations(groups)
            elif "query-args" in run_group:
                query_args = _generate_combinations(run_group["query-args"])
            else:
                query_args = []

            for arg_group in args:
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
                aargs = [_substitute_variables(arg, vs) for arg in aargs]
                definitions.append(Definition(
                    algorithm=name,
                    docker_tag=algo['docker-tag'],
                    module=algo['module'],
                    constructor=algo['constructor'],
                    arguments=aargs,
                    query_argument_groups=query_args,
                    disabled=algo.get('disabled', False)
                ))

    return definitions
