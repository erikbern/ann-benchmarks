import argparse
import json
from ann_benchmarks.datasets import DATASETS
from ann_benchmarks.algorithms.definitions import Definition, instantiate_algorithm
from ann_benchmarks.runner import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--module',
        required=True)
    parser.add_argument(
        '--constructor',
        required=True)
    parser.add_argument(
        '--count',
        required=True,
        type=int)
    parser.add_argument(
        '--json-args',
        action='store_true')
    parser.add_argument(
        '-a', '--arg',
        dest='args', action='append')
    args = parser.parse_args()
    if args.json_args:
        algo_args = [json.loads(arg) for arg in args.args]
    else:
        algo_args = args.args

    definition = Definition(
        algorithm=None, # not needed
        docker_tag=None, # also not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args
    )
    run(definition, args.dataset, args.count)
