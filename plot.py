import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.algorithms.definitions import get_definitions
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (get_plot_label, compute_metrics,
                                           create_linestyles, create_pointset)
from ann_benchmarks.results import (store_results, load_all_results,
                                    get_unique_algorithms, get_algorithm_name)


def create_plot(all_data, raw, x_log, y_log, xn, yn, fn_out, linestyles,
                batch):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        color, faded, linestyle, marker = linestyles[algo]
        handle, = plt.plot(xs, ys, '-', label=algo, color=color,
                           ms=7, mew=3, lw=3, linestyle=linestyle,
                           marker=marker)
        handles.append(handle)
        if raw:
            handle2, = plt.plot(axs, ays, '-', label=algo, color=faded,
                                ms=5, mew=2, lw=2, linestyle=linestyle,
                                marker=marker)
        labels.append(get_algorithm_name(algo, batch))

    if x_log:
        plt.gca().set_xscale('log')
    if y_log:
        plt.gca().set_yscale('log')
    plt.gca().set_title(get_plot_label(xm, ym))
    plt.gca().set_ylabel(ym['description'])
    plt.gca().set_xlabel(xm['description'])
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left',
                     bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
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
        default='glove-100-angular')
    parser.add_argument(
        '--count',
        default=10)
    parser.add_argument(
        '--definitions',
        metavar='FILE',
        help='load algorithm definitions from FILE',
        default='algos.yaml')
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '-o', '--output')
    parser.add_argument(
        '-x', '--x-axis',
        help='Which metric to use on the X-axis',
        choices=metrics.keys(),
        default="k-nn")
    parser.add_argument(
        '-y', '--y-axis',
        help='Which metric to use on the Y-axis',
        choices=metrics.keys(),
        default="qps")
    parser.add_argument(
        '-X', '--x-log',
        help='Draw the X-axis using a logarithmic scale',
        action='store_true')
    parser.add_argument(
        '-Y', '--y-log',
        help='Draw the Y-axis using a logarithmic scale',
        action='store_true')
    parser.add_argument(
        '--raw',
        help='Show raw results (not just Pareto frontier) in faded colours',
        action='store_true')
    parser.add_argument(
        '--batch',
        help='Plot runs in batch mode',
        action='store_true')
    parser.add_argument(
        '--recompute',
        help='Clears the cache and recomputes the metrics',
        action='store_true')
    args = parser.parse_args()

    if not args.output:
        args.output = 'results/%s.png' % get_algorithm_name(
            args.dataset, args.batch)
        print('writing output to %s' % args.output)

    dataset = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, True, args.batch)
    linestyles = create_linestyles(sorted(unique_algorithms))
    runs = compute_metrics(np.array(dataset["distances"]),
                           results, args.x_axis, args.y_axis, args.recompute)
    if not runs:
        raise Exception('Nothing to plot')

    create_plot(runs, args.raw, args.x_log,
                args.y_log, args.x_axis, args.y_axis, args.output,
                linestyles, args.batch)
