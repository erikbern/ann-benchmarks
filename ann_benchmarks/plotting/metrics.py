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
    return total_candidate_distance / total_closest_distance

all_metrics = {
    "k-nn": {
        "description": "10-NN precision - larger is better",
        "function": knn,
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "epsilon": {
        "description": "(epsilon)",
        "function": epsilon,
        "worst": float("-inf")
    },
    "rel": {
        "description": "(rel)",
        "function": rel,
        "worst": float("inf")
    }
}
