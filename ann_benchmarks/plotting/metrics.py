from __future__ import absolute_import

def knn(queries, run, epsilon=1e-10):
    results = zip(queries, run["results"])
    total = 0
    actual = 0
    for (query, max_distance, closest), [time, candidates] in results:
        within = filter(lambda (index, distance): \
                        distance <= (max_distance + epsilon), candidates)
        total += len(closest)
        actual += len(within)
    return float(actual) / float(total)

def epsilon(queries, run, epsilon=0.01):
    results = zip(queries, run["results"])
    total = 0
    actual = 0
    for (query, max_distance, closest), [time, candidates] in results:
        within = filter(lambda (index, distance): \
                        distance <= ((1 + epsilon) * max_distance), candidates)
        total += len(closest)
        actual += len(within)
    return float(actual) / float(total)

def rel(queries, run):
    results = zip(queries, run["results"])
    total_closest_distance = 0.0
    total_candidate_distance = 0.0
    for (query, max_distance, closest), [time, candidates] in results:
        for (ridx, rdist), (cidx, cdist) in zip(closest, candidates):
            total_closest_distance += rdist
            total_candidate_distance += cdist
    if total_closest_distance < 0.01:
        return float("inf")
    return total_candidate_distance / total_closest_distance

def queries_per_second(queries, run):
    return 1.0 / run["best_search_time"]

def index_size(queries, run):
    return run["index_size"]

def build_time(queries, run):
    return run["build_time"]

def candidates(queries, run):
    return run["candidates"]

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
