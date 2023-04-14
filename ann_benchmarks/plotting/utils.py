from __future__ import absolute_import

import itertools

import numpy as np

from ann_benchmarks.plotting.metrics import all_metrics as metrics


def get_or_create_metrics(run):
    if "metrics" not in run:
        run.create_group("metrics")
    return run["metrics"]


def create_pointset(data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    axs, ays, als = [], [], []
    # Generate Pareto frontier
    xs, ys, ls = [], [], []
    last_x = xm["worst"]
    comparator = (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
    for algo, algo_name, xv, yv in data:
        if not xv or not yv:
            continue
        axs.append(xv)
        ays.append(yv)
        als.append(algo_name)
        if comparator(xv, last_x):
            last_x = xv
            xs.append(xv)
            ys.append(yv)
            ls.append(algo_name)
    return xs, ys, ls, axs, ays, als


def compute_metrics(true_nn_distances, res, metric_1, metric_2, recompute=False):
    all_results = {}
    for i, (properties, run) in enumerate(res):
        algo = properties["algo"]
        algo_name = properties["name"]
        # cache distances to avoid access to hdf5 file
        run_distances = np.array(run["distances"])
        # cache times to avoid access to hdf5 file
        times = np.array(run["times"])
        if recompute and "metrics" in run:
            del run["metrics"]
        metrics_cache = get_or_create_metrics(run)

        metric_1_value = metrics[metric_1]["function"](
            true_nn_distances, run_distances, metrics_cache, times, properties
        )
        metric_2_value = metrics[metric_2]["function"](
            true_nn_distances, run_distances, metrics_cache, times, properties
        )

        print("%3d: %80s %12.3f %12.3f" % (i, algo_name, metric_1_value, metric_2_value))

        all_results.setdefault(algo, []).append((algo, algo_name, metric_1_value, metric_2_value))

    return all_results


def compute_all_metrics(true_nn_distances, run, properties, recompute=False):
    algo = properties["algo"]
    algo_name = properties["name"]
    print("--")
    print(algo_name)
    results = {}
    # cache distances to avoid access to hdf5 file
    run_distances = np.array(run["distances"])
    # cache times to avoid access to hdf5 file
    times = np.array(run["times"])
    if recompute and "metrics" in run:
        del run["metrics"]
    metrics_cache = get_or_create_metrics(run)

    for name, metric in metrics.items():
        v = metric["function"](true_nn_distances, run_distances, metrics_cache, times, properties)
        results[name] = v
        if v:
            print("%s: %g" % (name, v))
    return (algo, algo_name, results)


def compute_metrics_all_runs(dataset, res, recompute=False):
    true_nn_distances = list(dataset["distances"])
    for i, (properties, run) in enumerate(res):
        algo = properties["algo"]
        algo_name = properties["name"]
        # cache distances to avoid access to hdf5 file
        # print('Load distances and times')
        run_distances = np.array(run["distances"])
        times = np.array(run["times"])
        # print('... done')
        if recompute and "metrics" in run:
            print("Recomputing metrics, clearing cache")
            del run["metrics"]
        metrics_cache = get_or_create_metrics(run)

        dataset = properties["dataset"]

        run_result = {"algorithm": algo, "parameters": algo_name, "count": properties["count"]}
        for name, metric in metrics.items():
            v = metric["function"](true_nn_distances, run_distances, metrics_cache, times, properties)
            run_result[name] = v
        yield run_result


def generate_n_colors(n):
    vs = np.linspace(0.3, 0.9, 7)
    colors = [(0.9, 0.4, 0.4, 1.0)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        new_color = max(itertools.product(vs, vs, vs), key=lambda a: min(euclidean(a, b) for b in colors))
        colors.append(new_color + (1.0,))
    return colors


def create_linestyles(unique_algorithms):
    colors = dict(zip(unique_algorithms, generate_n_colors(len(unique_algorithms))))
    linestyles = dict((algo, ["--", "-.", "-", ":"][i % 4]) for i, algo in enumerate(unique_algorithms))
    markerstyles = dict((algo, ["+", "<", "o", "*", "x"][i % 5]) for i, algo in enumerate(unique_algorithms))
    faded = dict((algo, (r, g, b, 0.3)) for algo, (r, g, b, a) in colors.items())
    return dict((algo, (colors[algo], faded[algo], linestyles[algo], markerstyles[algo])) for algo in unique_algorithms)


def get_up_down(metric):
    if metric["worst"] == float("inf"):
        return "down"
    return "up"


def get_left_right(metric):
    if metric["worst"] == float("inf"):
        return "left"
    return "right"


def get_plot_label(xm, ym):
    template = "%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and" " to the %(leftright)s is better"
    return template % {
        "xlabel": xm["description"],
        "ylabel": ym["description"],
        "updown": get_up_down(ym),
        "leftright": get_left_right(xm),
    }
