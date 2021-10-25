import pandas as pd
import argparse

from ann_benchmarks.datasets import DATASETS, get_dataset
from ann_benchmarks.plotting.utils  import compute_metrics_all_runs
from ann_benchmarks.results import load_all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        help='Path to the output file',
        required=True)
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Path to the output csv file')
    args = parser.parse_args()

    datasets = DATASETS.keys()
    dfs = []

    for dataset_name in datasets:
        print("Looking at dataset", dataset_name)
        if len(list(load_all_results(dataset_name))) > 0:
            results = load_all_results(dataset_name)
            dataset, _ = get_dataset(dataset_name)
            results = compute_metrics_all_runs(dataset, results, args.recompute)
            dfs.append(pd.DataFrame(results))
    if len(dfs) > 0:
        data = pd.concat(dfs)
        data.to_csv(args.output, index=False)

