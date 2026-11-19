#!/usr/bin/env python3
"""Run this branch's configurations through Zvec's independent QG entry point.

This is a native protocol, not ANN-Benchmarks' historical Python-call timer.
The source repo and executable are explicit, so branches cannot be mixed silently.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['check', 'prepare', 'build', 'measure'])
    p.add_argument('--zvec-source', required=True, type=Path)
    p.add_argument('--binary', type=Path)
    p.add_argument('--dataset', choices=['sift', 'gist', 'glove', 'all'], default='all')
    p.add_argument('--bench-root', required=True, type=Path)
    p.add_argument('--artifacts', required=True, type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = a.zvec_source.resolve()
    binary = a.binary.resolve() if a.binary else source / 'build-r7i/bin/qg_bench'
    env = dict(os.environ, BENCH_ROOT=str(a.bench_root.resolve()), ARTIFACT_ROOT=str(a.artifacts.resolve()),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    datasets = ['sift', 'gist', 'glove'] if a.dataset == 'all' else [a.dataset]
    for ds in datasets:
        config = root / 'benchmark-configs/r7i-new-api' / (ds + '.json')
        subprocess.run([sys.executable, str(source / 'tools/core/bench/qg_bench.py'), a.phase,
                        '--config', str(config), '--binary', str(binary)], env=env, check=True)


if __name__ == '__main__':
    main()
