from __future__ import absolute_import

import os, json, pickle
from ann_benchmarks.main import get_fn
from ann_benchmarks.results import get_results
from ann_benchmarks.plotting.metrics import all_metrics as metrics
import matplotlib.pyplot as plt
import numpy

def create_pointset(algo, all_data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    data = all_data[algo]
    rev = ym["worst"] < 0
    data.sort(key=lambda (a, n, rs): rs[yn], reverse=rev) # sort by y coordinate

    axs, ays, als = [], [], []
    # Generate Pareto frontier
    xs, ys, ls = [], [], []
    last_x = xm["worst"]
    comparator = \
      (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
    for algo, algo_name, results in data:
        xv, yv = (results[xn], results[yn])
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

def enumerate_query_caches(ds):
    for f in os.listdir("queries/"):
        if f.startswith(ds + "_") and f.endswith(".p"):
            yield "queries/" + f

def compute_metrics(qs, ds):
    all_results = {}
    all_algos = set()
    for run in ds:
        algo = run["library"]
        algo_name = run["name"]

        print "--"
        print algo_name
        results = {}
        for name, metric in metrics.items():
            v = metric["function"](qs, run)
            results[name] = v
            if v:
                print "%s: %g" % (name, v)

        all_algos.add(algo)
        if not algo in all_results:
            all_results[algo] = []
        all_results[algo].append((algo, algo_name, results))
    return (all_results, all_algos)

def create_linestyles(algos):
    colors = plt.cm.Set1(numpy.linspace(0, 1, len(algos)))
    faded = [[r, g, b, 0.3] for [r, g, b, a] in colors]
    linestyles = {}
    for i, algo in enumerate(algos):
        linestyles[algo] = (colors[i], faded[i], ['--', '-.', '-', ':'][i%4], ['+', '<', 'o', '*', 'x'][i%5])
    return linestyles

def get_up_down(metric):
    if metric["worst"] == float("inf"):
        return "down"
    return "up"

def get_left_right(metric):
    if metric["worst"] == float("inf"):
        return "left"
    return "right"

def get_plot_label(xm, ym):
    return "%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and to the %(leftright)s is better" % {
            "xlabel" : xm["description"], "ylabel" : ym["description"], "updown" : get_up_down(ym), "leftright" : get_left_right(xm) }

