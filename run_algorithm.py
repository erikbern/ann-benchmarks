import argparse
from ann_benchmarks.datasets import DATASETS
from ann_benchmarks.algorithms.constructors import available_constructors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--algorithm',
        required=True)
    parser.add_argument(
        '-a', '--arg',
        type=list, dest='args')
    args = parser.parse_args()
    available_constructors[args.algorithm](*args.args)
