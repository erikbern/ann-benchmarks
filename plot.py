import os, json, pickle
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils  import get_plot_label, load_results, create_linestyles, create_pointset


def create_plot(all_data, golden, raw, x_log, y_log, xn, yn, fn_out, linestyles):
    xm, ym = (metrics[xn], metrics[yn])
# Now generate each plot
    handles = []
    labels = []
    if golden:
        plt.figure(figsize=(7, 4.35))
    else:
        plt.figure(figsize=(7, 7))
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        xs, ys, axs, ays, ls = create_pointset(algo, all_data, xn, yn)
        color, faded, linestyle, marker = linestyles[algo]
        handle, = plt.plot(xs, ys, '-', label=algo, color=color, ms=5, mew=1, lw=2, linestyle=linestyle, marker=marker)
        handles.append(handle)
        if raw:
            handle2, = plt.plot(axs, ays, '-', label=algo, color=faded, ms=5, mew=1, lw=2, linestyle=linestyle, marker=marker)
        labels.append(algo)

    if x_log:
        plt.gca().set_xscale('log')
    if y_log:
        plt.gca().set_yscale('log')
    plt.gca().set_title(get_plot_label(xm, ym))
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

if __name__ == "__main__":
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
    parser.add_argument(
        '-G', '--golden',
        help='Use golden ratio as plotsize',
        action='store_true')
    parser.add_argument(
        '--raw',
        help='Also show raw results in faded colours',
        action='store_true')
    args = parser.parse_args()
    runs, all_algos = load_results([ds for ds, _ in args.dataset])
    linestyles = create_linestyles(all_algos)
    for ds, fn_out in args.dataset:
        all_data = runs[ds]
        create_plot(all_data, args.golden, args.raw, args.x_log,
                args.y_log, args.x_axis, args.y_axis, fn_out, linestyles)
