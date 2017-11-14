from __future__ import absolute_import

def knn(dataset, run, epsilon=1e-10):
    total = 0
    actual = 0
    count = int(run.attrs['candidates'])
    for true_distances, found_distances in zip(dataset['distances'], run['distances']):
        within = [d for d in found_distances[:count] if d <= true_distances[count] + epsilon]
        total += count
        actual += len(within)
    return float(actual) / float(total)

def epsilon(dataset, run, epsilon=0.01):
    total = 0
    actual = 0
    count = int(run.attrs['candidates'])
    for true_distances, found_distances in zip(dataset['distances'], run['distances']):
        within = [d for d in found_distances[:count] if d <= true_distances[count] * (1 + epsilon)]
        total += count
        actual += len(within)
    return float(actual) / float(total)

def rel(dataset, run):
    total_closest_distance = 0.0
    total_candidate_distance = 0.0
    count = int(run.attrs['candidates'])
    for true_distances, found_distances in zip(dataset['distances'], run['distances']):
        for rdist, cdist in zip(true_distances, found_distances):
            total_closest_distance += rdist
            total_candidate_distance += cdist
    if total_closest_distance < 0.01:
        return float("inf")
    return total_candidate_distance / total_closest_distance

def queries_per_second(queries, run):
    return 1.0 / run.attrs["best_search_time"]

def index_size(queries, run):
    return run.attrs["index_size"]

def build_time(queries, run):
    return run.attrs["build_time"]

def candidates(queries, run):
    return run.attrs["candidates"]

all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": knn,
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": epsilon,
        "worst": float("-inf")
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda a,b: epsilon(a, b, 0.1),
        "worst": float("-inf")
    },
    "rel": {
        "description": "Relative Error",
        "function": rel,
        "worst": float("inf")
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": queries_per_second,
        "worst": float("-inf")
    },
    "build": {
        "description": "Build time (s)",
        "function": build_time,
        "worst": float("inf")
    },
    "candidates" : {
        "description": "Candidates generated",
        "function": candidates,
        "worst": float("inf")
    },
    "indexsize" : {
        "description": "Index size (kB)",
        "function": index_size,
        "worst": float("inf")
    },
    "queriessize" : {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda a, b: index_size(a,b) / queries_per_second(a,b),
        "worst": float("inf")
    }
}
