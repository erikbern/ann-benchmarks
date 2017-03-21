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
    '-x', '--x-axis',
    help = 'Which metric to use on the X-axis',
    choices = metrics.keys(),
    default = "k-nn")
parser.add_argument(
    '-y', '--y-axis',
    help = 'Which metric to use on the Y-axis',
    choices = metrics.keys(),
    default = "qps")
parser.add_argument(
    '-X', '--x-log',
    help='Draw the X-axis using a logarithmic scale',
    action='store_true')
parser.add_argument(
    '-Y', '--y-log',
    help='Draw the Y-axis using a logarithmic scale',
    action='store_true')
args = parser.parse_args()

xm = metrics[args.x_axis]
ym = metrics[args.y_axis]

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
            xv = xm["function"](queries, run)
            yv = ym["function"](queries, run)
            print xv, yv
            if not xv or not yv:
                continue

            all_algos.add(algo)
            if not algo in runs[ds]:
                runs[ds][algo] = []
            runs[ds][algo].append((algo, algo_name, xv, yv))

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
        data.sort(key=lambda (a, n, xv, yv): yv, reverse=True) # sort by time

        # Plot Pareto frontier
        xs, ys = [], []
        last_x = xm["worst"]
        comparator = \
          (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
        for algo, algo_name, xv, yv in data:
            if comparator(xv, last_x):
                last_x = xv
                xs.append(xv)
                ys.append(yv)

        color, linestyle, marker = linestyles[algo]
        handle, = plt.plot(xs, ys, '-', label=algo, color=color, ms=5, mew=1, lw=2, linestyle=linestyle, marker=marker)
        handles.append(handle)
        labels.append(algo)

    if args.x_log:
        plt.gca().set_xscale('log')
    if args.y_log:
        plt.gca().set_yscale('log')
    plt.gca().set_title('Plot')
    plt.gca().set_ylabel(ym['description'])
    plt.gca().set_xlabel(xm['description'])
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    if 'lim' in xm:
        plt.xlim(xm['lim'])
    if 'lim' in ym:
        plt.ylim(ym['lim'])
    plt.savefig(fn_out, bbox_inches='tight')
