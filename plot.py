import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='results/glove.txt')
parser.add_argument('--output', default='results/glove.png')

args = parser.parse_args()

all_data = {}

for line in open(args.input):
    algo, algo_name, build_time, search_time, precision = line.strip().split('\t')
    all_data.setdefault(algo, []).append((algo_name, float(build_time), float(search_time), float(precision)))

colors = plt.cm.rainbow(numpy.linspace(0, 1, len(all_data)))
handles = []
labels = []

for algo, color in zip(all_data.keys(), colors):
    data = all_data[algo]
    data.sort(key=lambda t: t[-2]) # sort by time
    xs = [t[-2] for t in data]
    ys = [t[-1] for t in data]
    ls = [t[0] for t in data]
    plt.plot(xs, ys, 'o', label=algo, color=color, markersize=3)
    #for i, l in enumerate(ls):
    #    plt.annotate(ls[i], (xs[i], ys[i]), color='lightgray', fontsize=8)

    # Plot Pareto frontier
    xs, ys = [], []
    last_y = float('-inf')
    for t in data:
        y = t[-1]
        if y > last_y:
            last_y = y
            ys.append(t[-1])
            xs.append(t[-2])
    handle, = plt.plot(xs, ys, 'o-', label=algo, color=color)
    handles.append(handle)
    labels.append(algo)

plt.gca().set_xscale('log')
plt.gca().set_title('Precision-Performance tradeoff - up and to the left is better')
plt.gca().set_xlabel('Time per query (s) - lower is better')
plt.gca().set_ylabel('10-NN precision - higher is better')
plt.gca().legend(handles, labels, loc='upper left')
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.ylim([0, 1.03])
plt.savefig(args.output)

