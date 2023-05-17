import sys
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, NewType
from collections import defaultdict

AlgoModule = NewType('AlgoModule', str)
MetricType = Literal["bit", "float"]

@dataclass
class RunGroup:
    args: Any = field(default_factory=dict)
    arg_groups: List[Dict] = field(default_factory=list)
    query_args: List[List[str]] = field(default_factory=list)

@dataclass()
class Algorithm:
    docker_tag: str
    module: str
    constructor: str
    base_args: Dict = field(default_factory=dict)
    disabled: bool = False
    run_groups: Dict[str, RunGroup] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

@dataclass
class MetricType:
    algorithms: Dict[str, Algorithm] = field(default_factory=dict)

@dataclass
class Metric:
    metric_types: Dict[str, MetricType] = field(default_factory=dict)

@dataclass
class Data:
    float: Metric = field(default_factory=Metric)
    bit: Metric = field(default_factory=Metric)


@dataclass
class AlgorithmFile:
    # maps float.euclidean.Algorithm
    algos: Dict[str, Dict[str, Algorithm]] = field(default_factory=dict)

def replace_hyphens_in_keys(data):
    """Replaces hyphens in keys with underscores for a given dictionary."""
    return {k.replace('-', '_'): v for k, v in data.items()}

def convert_raw_data_to_dataclasses(raw_data: Dict[str, Any]) -> Data:
    """Converts the raw data (from Yaml) into the above dataclasses."""
    data = Data()
    for metric_name, metric_types in raw_data.items():
        metric = Metric()
        for metric_type_name, algorithms in metric_types.items():
            metric_type = MetricType()
            for algorithm_name, algorithm_info in algorithms.items():
                run_groups_params = algorithm_info.pop('run-groups') if algorithm_info.get('run-groups') is not None else {}
                run_groups = {name: RunGroup(**replace_hyphens_in_keys(info)) for name, info in run_groups_params.items()}
                algorithm = Algorithm(run_groups=run_groups, **replace_hyphens_in_keys(algorithm_info))
                metric_type.algorithms[algorithm_name] = algorithm
            metric.metric_types[metric_type_name] = metric_type
        metric.metric_types[metric_name] = metric
    return data


def add_algorithm_metrics(files: Dict[AlgoModule, Dict[str, Dict[str, AlgorithmFile]]], metric_type: MetricType, metric_dict: Dict[str, MetricType]):
    """
    Updates the mapping of algorithms to configurations for a given metric type and data.
    Process a given metric dictionary and update the 'files' dictionary.
    """
    for metric, metric_type in metric_dict.items():
        for name, algorithm in metric_type.algorithms.items():
            algorithm_name = algorithm.module.split(".")[-1]
            if files[algorithm_name].get(metric_type) is None:
                files[algorithm_name][metric_type] = {}

            if files[algorithm_name][metric_type].get(metric) is None:
                files[algorithm_name][metric_type][metric] = []

            output = algorithm.to_dict()
            output["name"] = name
            files[algorithm_name][metric_type][metric].append(output)


def config_write(module_name: str, content: Dict[str, Dict[str, AlgorithmFile]]) -> None:
    """For a given algorithm module, write the algorithm's config to file."""
    class CustomDumper(yaml.SafeDumper):
        def represent_list(self, data):
            ## Avoid use [[]] for base lists
            if len(data) > 0 and isinstance(data[0], dict) and "docker_tag" in data[0].keys():
                return super().represent_list(data)
            else:
                return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    CustomDumper.add_representer(list, CustomDumper.represent_list)
    
    with open(f"ann_benchmarks/algorithms/{module_name}/config.yml", 'w+') as cfg:
        yaml.dump(content, cfg, Dumper=CustomDumper, default_flow_style=False)


if __name__ == "__main__":
    try:
        raw_yaml = sys.argv[0] if len(sys.argv) > 1 else "algos.yaml"
        with open(raw_yaml, 'r') as stream:
            raw_data = yaml.safe_load(stream)
    except FileNotFoundError:
        print("The file 'algos.yaml' was not found.")
        exit(1)

    data = convert_raw_data_to_dataclasses(raw_data)
    files: Dict[str, Dict[str, Dict[str, AlgorithmFile]]] = defaultdict(dict)

    add_algorithm_metrics(files, 'bit', data.bit.metric_types)
    add_algorithm_metrics(files, 'float', data.float.metric_types)

    for module_name, file_dict in files.items():
        config_write(module_name, file_dict)