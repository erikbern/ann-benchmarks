import h5py
import os
import argparse
import json

def get_build_stats(dataset, algo, path):
    algo_result_directory = [path, dataset, "10", algo]
    algo_results = os.path.join(*algo_result_directory)
    print("reading results from: " + algo_results)
    if not os.path.isdir(algo_results):
        raise "Not a valid path for results"

    results = {}
    # r=root, d=directories, f = files
    for r, d, f in os.walk(algo_results):
        for file in f:
            if file.endswith(".hdf5"):
                read_file = h5py.File(os.path.join(algo_results,file), 'r')
                setting = file.split(".")[0]
                results[setting] = {"build_time": read_file.attrs["build_time"], "index_size": read_file.attrs["index_size"]}

    build_stats_json = f"{dataset}-{algo}-build-stats.json"
    with open(build_stats_json, "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="the dataset for the results", default="glove-25-angular")
    parser.add_argument("--algo", help="the algo for the results", default="opensearchknn")
    parser.add_argument("--path", help="the directory path for the results", default=None)
    args = parser.parse_args()
    get_build_stats(dataset = args.dataset, algo=args.algo, path=args.path)