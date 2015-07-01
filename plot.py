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

for algo, color in zip(sorted(all_data.keys()), colors):
    data = all_data[algo]
    data.sort(key=lambda t: t[-2]) # sort by time
    ys = [1.0 / t[-2] for t in data] # queries per second
    xs = [t[-1] for t in data]
    ls = [t[0] for t in data]
    plt.plot(xs, ys, 'x', label=algo, color=color, markersize=2)
    #for i, l in enumerate(ls):
    #    plt.annotate(ls[i], (xs[i], ys[i]), color='lightgray', fontsize=8)

    # Plot Pareto frontier
    xs, ys = [], []
    last_y = float('-inf')
    for t in data:
        y = t[-1]
        if y > last_y:
            last_y = y
            xs.append(t[-1])
            ys.append(1.0 / t[-2])
    handle, = plt.plot(xs, ys, 'x-', label=algo, color=color, markersize=3)
    handles.append(handle)
    labels.append(algo)

plt.gca().set_yscale('log')
plt.gca().set_title('Precision-Performance tradeoff - up and to the right is better')
plt.gca().set_ylabel('Queries per second ($s^{-1}$) - larger is better')
plt.gca().set_xlabel('10-NN precision - larger is better')
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.xlim([0.0, 1.03])
plt.savefig(args.output, bbox_inches='tight')

