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
    plt.plot(xs, ys, 'x-', )
    for i, l in enumerate(ls):
        plt.annotate(ls[i], (xs[i], ys[i]))



plt.gca().set_xscale('log')     

plt.show()

                                       

    
