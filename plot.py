import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.results import get_results
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils  import get_plot_label, compute_metrics, create_linestyles, create_pointset


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
        xs, ys, ls, axs, ays, als = create_pointset(algo, all_data, xn, yn)
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
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        required=True)
    parser.add_argument(
        '--count',
        default=10)
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '-o', '--output',
        required=True)
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

    dataset = get_dataset(args.dataset)
    distance = dataset.attrs['distance']
    runs, all_algos = compute_metrics(dataset, get_results(
        args.dataset, args.count, distance))
    linestyles = create_linestyles(all_algos)

    create_plot(runs, args.golden, args.raw, args.x_log,
            args.y_log, args.x_axis, args.y_axis, args.output, linestyles)
