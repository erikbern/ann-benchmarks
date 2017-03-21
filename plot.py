import os, json, pickle
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

from ann_benchmarks.main import get_fn
from ann_benchmarks.plotting.metrics import all_metrics as metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    nargs = 2,
    metavar = ("DATASET", "OUTPUT"),
    action='append')
parser.add_argument(
    '--precision',
    help = 'Which precision metric to use',
    choices = metrics.keys(),
    default = metrics.keys()[0])
args = parser.parse_args()

runs = {}
all_algos = set()
for ds, _ in args.dataset:
    results_fn = get_fn("results", ds)
    queries_fn = get_fn("queries", ds)
    assert os.path.exists(queries_fn), """\
the queries file '%s' is missing""" % queries_fn

    queries = pickle.load(open(queries_fn))
    runs[ds] = {}
    with open(get_fn("results", ds)) as f:
        for line in f:
            run = json.loads(line)
            algo = run["library"]
            algo_name = run["name"]
            build_time = run["build_time"]
            search_time = run["best_search_time"]

            print "--"
            print algo_name
            precision = metrics[args.precision]["function"](queries, run)
            print precision
            if not precision:
                continue

            all_algos.add(algo)
            if not algo in runs[ds]:
                runs[ds][algo] = []
            runs[ds][algo].append(
                    (algo, algo_name, build_time, search_time, precision))

# Construct palette from the algorithm list
colors = plt.cm.Set1(numpy.linspace(0, 1, len(all_algos)))
linestyles = {}
for i, algo in enumerate(all_algos):
    linestyles[algo] = (colors[i], ['--', '-.', '-', ':'][i%4], ['+', '<', 'o', 'D', '*', 'x', 's'][i%7])

# Now generate each plot
for ds, fn_out in args.dataset:
    all_data = runs[ds]

    handles = []
    labels = []

    plt.figure(figsize=(7, 7))
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        data = all_data[algo]
        data.sort(key=lambda t: t[-2]) # sort by time
        ys = [1.0 / t[-2] for t in data] # queries per second
        xs = [t[-1] for t in data]
        ls = [t[0] for t in data]

        # Plot Pareto frontier
        xs, ys = [], []
        last_p = metrics[args.precision]["worst"]
        for algo, algo_name, build_time, search_time, precision in data:
            if precision > last_p:
                last_p = precision
                xs.append(precision)
                ys.append(1.0 / search_time)

        color, linestyle, marker = linestyles[algo]
        handle, = plt.plot(xs, ys, '-', label=algo, color=color, ms=5, mew=1, lw=2, linestyle=linestyle, marker=marker)
        handles.append(handle)
        labels.append(algo)

    plt.gca().set_yscale('log')
    plt.gca().set_title('Precision-Performance tradeoff - up and to the right is better')
    plt.gca().set_ylabel('Queries per second ($s^{-1}$) - larger is better')
    plt.gca().set_xlabel(metrics[args.precision]["description"])
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    if "lim" in metrics[args.precision]:
        plt.xlim(metrics[args.precision]["lim"])
    plt.savefig(fn_out, bbox_inches='tight')
