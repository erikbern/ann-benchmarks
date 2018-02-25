from __future__ import absolute_import

def knn(dataset_distances, run_distances, count, epsilon=1e-10):
    total = len(run_distances) * count
    actual = 0
    for true_distances, found_distances in zip(dataset_distances, run_distances):
        within = [d for d in found_distances[:count] if d <= true_distances[count - 1] + epsilon]
        actual += len(within)
    return float(actual) / float(total)

def epsilon(dataset_distances, run_distances, count, epsilon=0.01):
    total = len(run_distances) * count
    actual = 0
    for true_distances, found_distances in zip(dataset_distances, run_distances):
        within = [d for d in found_distances[:count] if d <= true_distances[count - 1] * (1 + epsilon)]
        actual += len(within)
    return float(actual) / float(total)

def rel(dataset_distances, run_distances):
    total_closest_distance = 0.0
    total_candidate_distance = 0.0
    for true_distances, found_distances in zip(dataset_distances, run_distances):
        for rdist, cdist in zip(true_distances, found_distances):
            total_closest_distance += rdist
            total_candidate_distance += cdist
    if total_closest_distance < 0.01:
        return float("inf")
    return total_candidate_distance / total_closest_distance

def queries_per_second(queries, attrs):
    return 1.0 / attrs["best_search_time"]

def index_size(queries, attrs):
    # TODO(erikbern): should replace this with peak memory usage or something
    return attrs.get("index_size", 0)

def build_time(queries, attrs):
    return attrs["build_time"]

def candidates(queries, attrs):
    return attrs["candidates"]

all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, run_attrs: knn(true_distances, run_distances, run_attrs["count"]),
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": lambda true_distances, run_distances, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"]),
        "worst": float("-inf")
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda true_distances, run_distances, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], 0.1),
        "worst": float("-inf")
    },
    "rel": {
        "description": "Relative Error",
        "function": lambda true_distances, run_distances, run_attrs: rel(true_distances, run_distances),
        "worst": float("inf")
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, run_attrs: queries_per_second(true_distances, run_attrs),
        "worst": float("-inf")
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_distances, run_distances, run_attrs: build_time(true_distances, run_attrs),
        "worst": float("inf")
    },
    "candidates" : {
        "description": "Candidates generated",
        "function": lambda true_distances, run_distances, run_attrs: candidates(true_distances, run_attrs),
        "worst": float("inf")
    },
    "indexsize" : {
        "description": "Index size (kB)",
        "function": lambda true_distances, run_distances, run_attrs: index_size(true_distances, run_attrs),
        "worst": float("inf")
    },
    "queriessize" : {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda true_distances, run_distances, run_attrs: index_size(true_distances, run_attrs) / queries_per_second(true_distances, run_attrs),
        "worst": float("inf")
    }
}
