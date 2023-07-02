import collections
import importlib
import os
import glob
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Optional


import yaml

Definition = collections.namedtuple(
    "Definition", ["algorithm", "constructor", "module", "docker_tag", "arguments", "query_argument_groups", "disabled"]
)


def instantiate_algorithm(definition: Definition):
    """Creates a `BaseANN` object for a specific algorithm definition.
     
    The constructor for the algorithm definition is, generally, located at ann_benchmarks/algorithms/*/module.py.
    """
    print("Trying to instantiate %s.%s(%s)" % (definition.module, definition.constructor, definition.arguments))
    module = importlib.import_module(f"{definition.module}.module")
    constructor = getattr(module, definition.constructor)
    return constructor(*definition.arguments)


class InstantiationStatus(Enum):
    AVAILABLE = 0
    NO_CONSTRUCTOR = 1
    NO_MODULE = 2


def algorithm_status(definition):
    try:
        module = importlib.import_module(definition.module + '.module')
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
        return dict([(k, _substitute_variables(v, vs)) for k, v in arg.items()])
    elif isinstance(arg, list):
        return [_substitute_variables(a, vs) for a in arg]
    elif isinstance(arg, str) and arg in vs:
        return vs[arg]
    else:
        return arg

def get_config_files(base_dir: str = "ann_benchmarks/algorithms") -> List[str]:
    """Get config files for all algorithms."""
    config_files = glob.glob(os.path.join(base_dir, "*", "config.yml"))
    return list(
        set(config_files) - {f"{base_dir}/base/config.yml"}
    )


def load_configs(point_type: str, base_dir: str = "ann_benchmarks/algorithms") -> Dict[str, Any]:
    """Load algorithm configurations for a given point_type."""
    config_files = get_config_files(base_dir=base_dir)
    configs = {}
    for config_file in config_files:
        with open(config_file, 'r') as stream:
            try:
                config_data = yaml.safe_load(stream)
                algorithm_name = os.path.basename(os.path.dirname(config_file))
                if point_type in config_data:
                    configs[algorithm_name] = config_data[point_type]
            except yaml.YAMLError as e:
                print(f"Error loading YAML from {config_file}: {e}")
    return configs

def _get_definitions(base_dir: str = "ann_benchmarks/algorithms"):
    """Load algorithm configurations for a given point_type."""
    config_files = get_config_files(base_dir=base_dir)
    configs = {}
    for config_file in config_files:
        with open(config_file, 'r') as stream:
            try:
                config_data = yaml.safe_load(stream)
                algorithm_name = os.path.basename(os.path.dirname(config_file))
                configs[algorithm_name] = config_data
            except yaml.YAMLError as e:
                print(f"Error loading YAML from {config_file}: {e}")
    return configs

def _get_algorithm_definitions(point_type: str, distance_metric: str) -> Dict[str, Dict[str, Any]]:
    """Get algorithm definitions for a specific point type and distance metric.
    
    A specific algorithm folder can have multiple algorithm definitions for a given point type and 
    metric. For example, `ann_benchmarks.algorithms.nmslib` has two definitions for euclidean float
    data: specifically `SW-graph(nmslib)` and `hnsw(nmslib)`, even though the module is named nmslib.

    If an algorithm has an 'any' distance metric is found for the specific point type, it is used 
    regardless (and takes precendence) over if the distance metric is present.

    Returns: A mapping from the algorithm name (not the algorithm class), to the algorithm definitions, i.e.:
    ```
    {
        'SW-graph(nmslib)': {
            "base_args": ['@metric', hnsw],
            "constructor": NmslibReuseIndex,
            "disabled": false,
            "docker_tag": ann-benchmarks-nmslib,
            ...
        }, 
        'SW-graph(nmslib)': {
            "base_args": ['@metric', sw-graph],
            "constructor": NmslibReuseIndex,
            "disabled": false,
            "docker_tag": ann-benchmarks-nmslib,
            ...
        }
    }
    ```
    """
    configs = load_configs(point_type)
    definitions = {}

    # param `_` is filename, not specific name
    for _, config in configs.items():
        c = []
        if "any" in config: # "any" branch must come first
            c = config["any"]
        elif distance_metric in config:
            c = config[distance_metric]

        for cc in c:
            definitions[cc.pop("name")] = cc

    return definitions

def list_algorithms(base_dir: str = "ann_benchmarks/algorithms") -> None:
    """Output a list of all algorithms, with their supported point types and metrics."""
    definitions = _get_definitions(base_dir)

    print("The following algorithms are supported...", definitions)
    for algorithm in definitions:
        print('\t... for the algorithm "%s"...' % algorithm)
        for point_type in definitions[algorithm]:
            print('\t\t... and the point type "%s", metrics: ' % point_type)
            for metric in definitions[algorithm][point_type]:
                print("\t\t\t%s" % metric)

def create_definitions_from_algorithm(name: str, algo: Dict[str, Any], dimension, point_type="float", distance_metric="euclidean", count=10) -> List[Definition]:
    definitions = []
    for k in ["docker_tag", "module", "constructor"]:
            if k not in algo:
                raise Exception('algorithm %s does not define a "%s" property' % (name, k))

    base_args = []
    if "base_args" in algo:
        base_args = algo["base_args"]

    for run_group in algo["run_groups"].values():
        if "arg_groups" in run_group:
            groups = []
            for arg_group in run_group["arg_groups"]:
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

        if "query_arg_groups" in run_group:
            groups = []
            for arg_group in run_group["query_arg_groups"]:
                if isinstance(arg_group, dict):
                    groups.append(_generate_combinations(arg_group))
                elif len(arg_group) > 0:
                    groups.append(arg_group)
            query_args = _generate_combinations(groups)
        elif "query_args" in run_group:
            query_args = _generate_combinations(run_group["query_args"])
        else:
            query_args = []
        
        for arg_group in args:
            aargs = []
            aargs.extend(base_args)
            if isinstance(arg_group, list):
                aargs.extend(arg_group)
            else:
                aargs.append(arg_group)

            vs = {"@count": count, "@metric": distance_metric, "@dimension": dimension}
            aargs = [_substitute_variables(arg, vs) for arg in aargs]
            definitions.append(
                Definition(
                    algorithm=name,
                    docker_tag=algo["docker_tag"],
                    module=algo["module"],
                    constructor=algo["constructor"],
                    arguments=aargs,
                    query_argument_groups=query_args,
                    disabled=algo.get("disabled", False),
                )
            )
    return definitions

def get_definitions(definition_file, dimension, point_type="float", distance_metric="euclidean", count=10) -> List[Definition]:
    algorithm_definitions = _get_algorithm_definitions(point_type=point_type,  distance_metric=distance_metric)

    definitions: List[Definition] = []

    # Map this for each config.yml
    for (name, algo) in algorithm_definitions.items():
        definitions.extend(
            create_definitions_from_algorithm(name, algo, dimension, point_type, distance_metric, count)
        )
        

    return definitions
