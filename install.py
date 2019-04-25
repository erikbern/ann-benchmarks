import json
import os
import argparse
import subprocess
from multiprocessing import Pool
from ann_benchmarks.main import positive_int


def build(library, args):
    print('Building %s...' % library)
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""
    subprocess.check_call(
        'docker build %s --rm -t ann-benchmarks-%s -f'
        ' install/Dockerfile.%s .' % (q, library, library), shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--proc",
        default=1,
        type=positive_int,
        help="the number of process to build docker images")
    parser.add_argument(
        '--algorithm',
        metavar='NAME',
        help='build only the named algorithm image',
        default=None)
    parser.add_argument(
        '--build-arg',
        help='pass given args to all docker builds',
        nargs="+")
    args = parser.parse_args()

    print('Building base image...')
    subprocess.check_call(
        'docker build \
        --rm -t ann-benchmarks -f install/Dockerfile .', shell=True)

    if args.algorithm:
        print('Building algorithm(%s) image...' % args.algorithm)
        build(args.algorithm, args.build_arg)
    elif os.getenv('LIBRARY'):
        print('Building algorithm(%s) image...' % os.getenv('LIBRARY'))
        build(os.getenv('LIBRARY'), args.build_arg)
    else:
        print('Building algorithm images... with (%d) processes' % args.proc)
        dockerfiles = []
        for fn in os.listdir('install'):
            if fn.startswith('Dockerfile.'):
                dockerfiles.append(fn.split('.')[-1])

        if args.proc == 1:
            [build(tag, args.build_arg) for tag in dockerfiles]
        else:
            pool = Pool(processes=args.proc)
            pool.map(lambda x: build(x, args.build_arg), dockerfiles)
            pool.close()
            pool.join()
