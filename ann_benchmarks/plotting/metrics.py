from __future__ import absolute_import

def knn(dataset_distances, run_distances, count, epsilon=1e-10):
    total = 0
    actual = 0
    for true_distances, found_distances in zip(dataset_distances, run_distances):
        within = [d for d in found_distances[:count] if d <= true_distances[count - 1] + epsilon]
        total += count
        actual += len(within)
    return float(actual) / float(total)

def epsilon(dataset_distances, run_distances, count, epsilon=0.01):
    total = 0
    actual = 0
    for true_distances, found_distances in zip(dataset_distances, run_distances):
        within = [d for d in found_distances[:count] if d <= true_distances[count - 1] * (1 + epsilon)]
        total += count
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
        "function": lambda dataset, run, count: knn(dataset['distances'], run['distances'],
            count),
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": lambda dataset, run, count: epsilon(dataset['distances'], run['distances'],
            count),
        "worst": float("-inf")
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda dataset, run, count: epsilon(dataset['distances'], run['distances'],
            count, 0.1),
        "worst": float("-inf")
    },
    "rel": {
        "description": "Relative Error",
        "function": lambda dataset, run, count: rel(dataset['distances'], run['distances']),
        "worst": float("inf")
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda dataset, run, count: queries_per_second(dataset, run.attrs),
        "worst": float("-inf")
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda dataset, run, count: build_time(dataset, run.attrs),
        "worst": float("inf")
    },
    "candidates" : {
        "description": "Candidates generated",
        "function": lambda dataset, run, count: candidates(dataset, run.attrs),
        "worst": float("inf")
    },
    "indexsize" : {
        "description": "Index size (kB)",
        "function": lambda dataset, run, count: index_size(dataset, run.attrs),
        "worst": float("inf")
    },
    "queriessize" : {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda dataset, run, count: index_size(dataset, run.attrs) / queries_per_second(dataset, run.attrs),
        "worst": float("inf")
    }
}
