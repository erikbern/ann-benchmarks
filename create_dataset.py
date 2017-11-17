import argparse
from ann_benchmarks.datasets import DATASETS, get_dataset_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    args = parser.parse_args()
    fn = get_dataset_fn(args.dataset)
    DATASETS[args.dataset](fn)
