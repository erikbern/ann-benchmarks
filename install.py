import json
import os
import argparse
import subprocess
from multiprocessing import Pool
from ann_benchmarks.main import positive_int


def build(library):
    print('Building %s...' % library)
    subprocess.check_call(
        'docker build \
        --rm -t ann-benchmarks-%s -f install/Dockerfile.%s .' % (library, library), shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--proc",
        default=4,
        type=positive_int,
        help="the number of process to build docker images")
    parser.add_argument(
        '--algorithm',
        metavar='NAME',
        help='build only the named algorithm image',
        default=None)
    args = parser.parse_args()

    print('Building base image...')
    subprocess.check_call(
        'docker build \
        --rm -t ann-benchmarks -f install/Dockerfile .', shell=True)

    if args.algorithm:
        print('Building algorithm(%s) image...' % args.algorithm)
        build(args.algorithm)
    elif os.getenv('LIBRARY'):
        print('Building algorithm(%s) image...' % os.getenv('LIBRARY'))
        build(os.getenv('LIBRARY'))
    else:
        print('Building algorithm images... with (%d) processes' % args.proc)
        dockerfiles = []
        for fn in os.listdir('install'):
            if fn.startswith('Dockerfile.'):
                dockerfiles.append(fn.split('.')[-1])

        pool = Pool(processes=args.proc)
        pool.map(build, dockerfiles)
        pool.close()
        pool.join()
