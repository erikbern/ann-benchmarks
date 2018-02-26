from __future__ import absolute_import
import argparse
import docker
import os
import random
import sys
import shutil
import traceback

from ann_benchmarks.datasets import get_dataset, DATASETS
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.definitions import get_definitions, list_algorithms, get_result_filename, algorithm_status, InstantiationStatus
from ann_benchmarks.runner import run, run_docker


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
        default='glove-100-angular',
        choices=DATASETS.keys())
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
        '--docker-tag',
        metavar='NAME',
        help='run only algorithms in a particular docker image',
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
        '--local',
        action='store_true',
        help='If set, then will run everything locally (inside the same process) rather than using Docker')
    parser.add_argument(
        '--max-n-algorithms',
        type=int,
        help='Max number of algorithms to run (just used for testing)',
        default=-1)

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None

    if hasattr(args, "list_algorithms"):
        list_algorithms(args.definitions)
        sys.exit(0)

    # Nmslib specific code
    # Remove old indices stored on disk
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    dataset = get_dataset(args.dataset)
    dimension = len(dataset['train'][0]) # TODO(erikbern): ugly
    point_type = 'float' # TODO(erikbern): should look at the type of X_train
    distance = dataset.attrs['distance']
    definitions = get_definitions(args.definitions, dimension, point_type, distance, args.count)

    # TODO(erikbern): should make this a helper function somewhere
    definitions = [definition for definition in definitions if not os.path.exists(get_result_filename(args.dataset, args.count, definition))]

    random.shuffle(definitions)
    
    if args.algorithm:
        print('running only', args.algorithm)
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    if not args.local:
        # See which Docker images we have available
        docker_client = docker.from_env()
        docker_tags = set()
        for image in docker_client.images.list():
            for tag in image.tags:
                tag, _ = tag.split(':')
                docker_tags.add(tag)

        if args.docker_tag:
            print('running only', args.docker_tag)
            definitions = [d for d in definitions if d.docker_tag == args.docker_tag]

        if set(d.docker_tag for d in definitions).difference(docker_tags):
            print('not all docker images available, only:', set(docker_tags))
            print('missing docker images:', set(d.docker_tag for d in definitions).difference(docker_tags))
            definitions = [d for d in definitions if d.docker_tag in docker_tags]
    else:
        def _test(df):
            status = algorithm_status(df)
            # If the module was loaded but doesn't actually have a constructor of
            # the right name, then the definition is broken
            assert status != InstantiationStatus.NO_CONSTRUCTOR, """\
%s.%s(%s): error: the module '%s' does not expose the named constructor""" % (df.module, df.constructor, df.arguments, df.module)
            if status == InstantiationStatus.NO_MODULE:
                # If the module couldn't be loaded (presumably because of a missing
                # dependency), print a warning and remove this definition from the
                # list of things to be run
                print("""\
%s.%s(%s): warning: the module '%s' could not be loaded; skipping""" % (df.module, df.constructor, df.arguments, df.module))
                return False
            else:
                return True
        definitions = [d for d in definitions if _test(d)]

    if args.max_n_algorithms >= 0:
        definitions = definitions[:args.max_n_algorithms]

    print('order:', definitions)

    for definition in definitions:
        print(definition, '...')

        try:
            if args.local:
                run(definition, args.dataset, args.count, args.runs)
            else:
                run_docker(definition, args.dataset, args.count, args.runs)
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
