from __future__ import absolute_import
import time, os, multiprocessing, argparse, resource, random
import sys
import shutil

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.results import get_results
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.definitions import get_algorithms, list_algorithms
from ann_benchmarks.runner import run


def positive_int(s):
    i = None
    try:
        i = int(s)
    except ValueError:
        pass
    if not i or i < 1:
        raise argparse.ArgumentTypeError("%r is not a positive integer" % s)
    return i

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--dataset',
            metavar='NAME',
            help='the dataset to load training points from',
            default='glove-100-angular')
    parser.add_argument(
            "-k", "--count",
            default=10,
            type=positive_int,
            help="the number of near neighbours to search for")
    parser.add_argument(
            '--definitions',
            metavar='FILE',
            help='load algorithm definitions from FILE',
            default='algos.yaml')
    parser.add_argument(
            '--algorithm',
            metavar='NAME',
            help='run only the named algorithm',
            default=None)
    parser.add_argument(
            '--sub-algorithm',
            metavar='NAME',
            help='run only the named instance of an algorithm (requires --algo)',
            default=None)
    parser.add_argument(
            '--list-algorithms',
            help='print the names of all known algorithms and exit',
            action='store_true',
            default=argparse.SUPPRESS)
    parser.add_argument(
            '--force',
            help='''re-run algorithms even if their results already exist''',
            action='store_true')
    parser.add_argument(
            '--runs',
            metavar='COUNT',
            type=positive_int,
            help='run each algorithm instance %(metavar)s times and use only the best result',
            default=3)
    parser.add_argument(
            '--timeout',
            type=int,
            help='Timeout (in seconds) for each individual algorithm run, or -1 if no timeout should be set',
            default=-1)
    parser.add_argument(
            '--single',
            help='run only a single algorithm instance at a time',
            action='store_true')
    parser.add_argument(
            '--batch',
            help='Provide Queryset as Batch',
            action='store_true')
    parser.add_argument(
            '--no_save_index',
            help='do not save indices',
            action='store_true')

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None

    if hasattr(args, "list_algorithms"):
        list_algorithms(args.definitions)
        sys.exit(0)

    # Set resource limits to prevent memory bombs
    memory_limit = 12 * 2**30
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if soft == resource.RLIM_INFINITY or soft >= memory_limit:
        print('resetting memory limit from', soft, 'to', memory_limit)
        resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))

    # Nmslib specific code
    # Remove old indices stored on disk
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    algos_already_run = set()
    if not args.force:
        for res in get_results(args.dataset, args.count):
            algos_already_run.add((res.attrs["library"], res.attrs["name"]))

    dataset = get_dataset(args.dataset)
    dimension = len(dataset['train'][0]) # TODO(erikbern): ugly
    point_type = 'float' # TODO(erikbern): should look at the type of X_train
    distance = dataset.attrs['distance']
    algos = get_algorithms(args.definitions, dimension, point_type, distance, args.count)

    if args.algorithm:
        print('running only', args.algorithm)
        algos = {args.algorithm: algos[args.algorithm]}
        if args.sub_algorithm:
            algos[args.algorithm] = \
              [algo for algo in algos[args.algorithm] if algo.name == args.sub_algorithm]

    algos_flat = []

    for library in algos.keys():
        for algo in algos[library]:
            if (library, algo.name) not in algos_already_run:
                algos_flat.append((library, algo))

    random.shuffle(algos_flat)

    print('order:', [a.name for l, a in algos_flat])

    for library, algo in algos_flat:
        print(algo.name, '...')
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(
            target=run,
            args=(args.dataset, args.count, library, algo,
                  args.runs, args.single, args.batch))

        # TODO(erikbern): this no longer handles timeouts etc but going to replace this with Docker shortly anyway
        p.start()
        p.join()
