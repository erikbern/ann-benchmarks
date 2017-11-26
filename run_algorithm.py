import argparse
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
        '--library')
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
        '-a', '--arg',
        dest='args', action='append')
    args = parser.parse_args()
    definition = Definition(
        algorithm=None, # not needed
        library=args.library,
        module=args.module,
        constructor=args.constructor,
        arguments=args.args
    )
    run(definition, args.dataset, args.count)
