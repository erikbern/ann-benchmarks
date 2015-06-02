import numpy
import matplotlib.pyplot as plt

all_data = {}

for line in open('data.tsv'):
    algo, algo_name, build_time, search_time, precision = line.strip().split('\t')
    all_data.setdefault(algo, []).append((algo_name, float(build_time), float(search_time), float(precision)))

for algo, data in all_data.iteritems():
    data.sort(key=lambda t: t[-2]) # sort by time
    xs = [t[-2] for t in data]
    ys = [t[-1] for t in data]
    ls = [t[0] for t in data]
    # plt.plot(xs, ys, 'x', label=algo)
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
    plt.plot(xs, ys, 'x-', label=algo)

plt.gca().set_xscale('log')
plt.gca().set_title('Precision-Performance tradeoff - up and to the left is better')
plt.gca().set_xlabel('Time per query (s) - lower is better')
plt.gca().set_ylabel('10-NN precision - higher is better')
plt.gca().legend()
plt.grid(b=True, which='both', color='0.65',linestyle='-')

plt.savefig('plot.png')


                                       

    
