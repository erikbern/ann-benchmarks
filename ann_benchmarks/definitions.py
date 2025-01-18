import collections
from dataclasses import dataclass
import importlib
import os
import glob
import logging
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Optional, Union

from ann_benchmarks.algorithms.base.module import BaseANN

import yaml

@dataclass
class Definition:
    algorithm: str
    constructor: str
    module: str
    docker_tag: str
    arguments: List[Any]
    query_argument_groups: List[List[Any]]
    disabled: bool


def instantiate_algorithm(definition: Definition) -> BaseANN:
    """
    Create a `BaseANN` from a definition.

    Args:
        definition (Definition): An object containing information about the algorithm.

    Returns:
        BaseANN: Instantiated algorithm

    Note:
        The constructors for the algorithm definition are generally located at
        ann_benchmarks/algorithms/*/module.py.
    """
    print(f"Trying to instantiate {definition.module}.{definition.constructor}({definition.arguments})")
    module = importlib.import_module(f"{definition.module}.module")
    constructor = getattr(module, definition.constructor)
    return constructor(*definition.arguments)


class InstantiationStatus(Enum):
    """Possible status of instantiating an algorithm from a python module import."""
    AVAILABLE = 0
    NO_CONSTRUCTOR = 1
    NO_MODULE = 2


def algorithm_status(definition: Definition) -> InstantiationStatus:
    """
    Determine the instantiation status of the algorithm based on its python module and constructor.

    Attempts to find the Python class constructor based on the definition's module path and
    constructor name.

    Args:
        definition (Definition): The algorithm definition containing module and constructor.

    Returns:
        InstantiationStatus: The status of the algorithm instantiation.
    """
    try:
        module = importlib.import_module(f"{definition.module}.module")
        if hasattr(module, definition.constructor):
            return InstantiationStatus.AVAILABLE
        else:
            return InstantiationStatus.NO_CONSTRUCTOR
    except ImportError:
        logging.exception("Could not import algorithm module for %s",
                          definition.module)
        return InstantiationStatus.NO_MODULE


def _generate_combinations(args: Union[List[Any], Dict[Any, Any]]) -> List[Union[List[Any], Dict[Any, Any]]]:
    """
    Generate combinations of elements from args, either the list or combinations of key-value pairs in a dict.

    Args:
        args (Union[List[Any], Dict[Any, Any]]): Input list or dict to generate combinations from.

    Returns:
        List[Union[List[Any], Dict[Any, Any]]]: List of combinations generated from input.

    Raises:
        TypeError: If input is neither a list nor a dict.
    """

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
        raise TypeError(f"No args handling exists for {type(args).__name__}")


def _substitute_variables(arg: Any, vs: Dict[str, Any]) -> Any:
    """
    Substitutes any string variables present in the argument structure with provided values.

    Support for nested substitution in the case `arg` is a List or Dict.

    Args:
        arg (Any): The argument structure, can be of type dict, list, or str.
        variable_substitutions (Dict[str, Any]): A mapping variable names to their values.

    Returns:
        Any: The argument structure with variables substituted by their corresponding values.
    """
    if isinstance(arg, dict):
        return {k: _substitute_variables(v, vs) for k, v in arg.items()}
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
        set(config_files) - {os.path.join(base_dir, "base", "config.yml")}
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

def _get_definitions(base_dir: str = "ann_benchmarks/algorithms") -> List[Dict[str, Any]]:
    """Load algorithm configurations."""
    config_files = get_config_files(base_dir=base_dir)
    configs = []
    for config_file in config_files:
        with open(config_file, 'r') as stream:
            try:
                config_data = yaml.safe_load(stream)
                configs.append(config_data)
            except yaml.YAMLError as e:
                print(f"Error loading YAML from {config_file}: {e}")
    return configs

def _get_algorithm_definitions(point_type: str, distance_metric: str, base_dir: str = "ann_benchmarks/algorithms") -> Dict[str, Dict[str, Any]]:
    """Get algorithm definitions for a specific point type and distance metric.

    A specific algorithm folder can have multiple algorithm definitions for a given point type and
    metric. For example, `ann_benchmarks.algorithms.nmslib` has two definitions for euclidean float
    data: specifically `SW-graph(nmslib)` and `hnsw(nmslib)`, even though the module is named nmslib.

    If an algorithm has an 'any' distance metric, it is also included.

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
    configs = load_configs(point_type, base_dir)
    definitions = {}

    # param `_` is filename, not specific name
    for _, config in configs.items():
        c = []
        if "any" in config:
            c.extend(config["any"])
        if distance_metric in config:
            c.extend(config[distance_metric])
        for cc in c:
            definitions[cc.pop("name")] = cc

    return definitions

def list_algorithms(base_dir: str = "ann_benchmarks/algorithms") -> None:
    """
    Output (to stdout), a list of all algorithms, with their supported point types and metrics.

    Args:
        base_dir (str, optional): The base directory where the algorithms are stored.
                                  Defaults to "ann_benchmarks/algorithms".
    """
    all_configs = _get_definitions(base_dir)
    data = {}
    for algo_configs in all_configs:
        for point_type, config_for_point_type in algo_configs.items():
            for metric, ccc in config_for_point_type.items():
                algo_name = ccc[0]["name"]
                if algo_name not in data:
                    data[algo_name] = {}
                if point_type not in data[algo_name]:
                    data[algo_name][point_type] = []
                data[algo_name][point_type].append(metric)

    print("The following algorithms are supported:", ", ".join(data))
    print("Details of supported metrics and data types: ")
    for algorithm in data:
        print('\t... for the algorithm "%s"...' % algorithm)

        for point_type in data[algorithm]:
            print('\t\t... and the point type "%s", metrics: ' % point_type)

            for metric in data[algorithm][point_type]:
                print("\t\t\t%s" % metric)


def generate_arg_combinations(run_group: Dict[str, Any], arg_type: str) -> List:
    """Generate combinations of arguments from a run group for a specific argument type.

    Args:
        run_group (Dict[str, Any]): The run group containing argument definitions.
        arg_type (str): The type of argument group to generate combinations for.

    Returns:
        List: A list of combinations of arguments.
    """
    if arg_type in ["arg_groups", "query_arg_groups"]:
        groups = []
        for arg_group in run_group[arg_type]:
            if isinstance(arg_group, dict):
                # Dictionaries need to be expanded into lists in order
                # for the subsequent call to _generate_combinations to
                # do the right thing
                groups.append(_generate_combinations(arg_group))
            else:
                groups.append(arg_group)
        return _generate_combinations(groups)
    elif arg_type in ["args", "query_args"]:
        return _generate_combinations(run_group[arg_type])
    else:
        return []


def prepare_args(run_group: Dict[str, Any]) -> List:
    """For an Algorithm's run group, prepare arguments.

    An `arg_groups` is preferenced over an `args` key.

    Args:
        run_group (Dict[str, Any]): The run group containing argument definitions.

    Returns:
        List: A list of prepared arguments.

    Raises:
        ValueError: If the structure of the run group is not recognized.
    """
    if "args" in run_group or "arg_groups" in run_group:
        return generate_arg_combinations(run_group, "arg_groups" if "arg_groups" in run_group else "args")
    else:
        raise ValueError(f"Unknown run_group structure: {run_group}")


def prepare_query_args(run_group: Dict[str, Any]) -> List:
    """For an algorithm's run group, prepare query args/ query arg groups.

    Args:
        run_group (Dict[str, Any]): The run group containing argument definitions.

    Returns:
        List: A list of prepared query arguments.
    """
    if "query_args" in run_group or "query_arg_groups" in run_group:
        return generate_arg_combinations(run_group, "query_arg_groups" if "query_arg_groups" in run_group else "query_args")
    else:
        return []


def create_definitions_from_algorithm(name: str, algo: Dict[str, Any], dimension: int, distance_metric: str = "euclidean", count: int = 10) -> List[Definition]:
    """
    Create definitions from an indvidual algorithm. An algorithm (e.g. annoy) can have multiple
     definitions based on various run groups (see config.ymls for clear examples).

    Args:
        name (str): Name of the algorithm.
        algo (Dict[str, Any]): Dictionary with algorithm parameters.
        dimension (int): Dimension of the algorithm.
        distance_metric (str, optional): Distance metric used by the algorithm. Defaults to "euclidean".
        count (int, optional): Count of the definitions to be created. Defaults to 10.

    Raises:
        Exception: If the algorithm does not define "docker_tag", "module" or "constructor" properties.

    Returns:
        List[Definition]: A list of definitions created from the algorithm.
    """
    required_properties = ["docker_tag", "module", "constructor"]
    missing_properties = [prop for prop in required_properties if prop not in algo]
    if missing_properties:
        raise ValueError(f"Algorithm {name} is missing the following properties: {', '.join(missing_properties)}")

    base_args = algo.get("base_args", [])

    definitions = []
    for run_group in algo["run_groups"].values():
        args = prepare_args(run_group)
        query_args = prepare_query_args(run_group)

        for arg_group in args:
            current_args = []
            current_args.extend(base_args)
            if isinstance(arg_group, list):
                current_args.extend(arg_group)
            else:
                current_args.append(arg_group)

            vs = {"@count": count, "@metric": distance_metric, "@dimension": dimension}
            current_args = [_substitute_variables(arg, vs) for arg in current_args]

            definitions.append(
                Definition(
                    algorithm=name,
                    docker_tag=algo["docker_tag"],
                    module=algo["module"],
                    constructor=algo["constructor"],
                    arguments=current_args,
                    query_argument_groups=query_args,
                    disabled=algo.get("disabled", False),
                )
            )
    return definitions

def get_definitions(
    dimension: int,
    point_type: str = "float",
    distance_metric: str = "euclidean",
    count: int = 10,
    base_dir: str = "ann_benchmarks/algorithms"
) -> List[Definition]:
    algorithm_definitions = _get_algorithm_definitions(point_type=point_type,
                                                       distance_metric=distance_metric,
                                                       base_dir=base_dir
                                                       )

    definitions: List[Definition] = []

    # Map this for each config.yml
    for (name, algo) in algorithm_definitions.items():
        definitions.extend(
            create_definitions_from_algorithm(name, algo, dimension, distance_metric, count)
        )


    return definitions
