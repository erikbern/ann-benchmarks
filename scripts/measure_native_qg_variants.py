#!/usr/bin/env python3
"""Measure both already-built QG variants serially under one isolation window."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--succinct', required=True, type=Path)
p.add_argument('--exploration', required=True, type=Path)
p.add_argument('--bench-root', required=True, type=Path)
p.add_argument('--artifacts', required=True, type=Path)
a = p.parse_args()
for name, source in [('succinct', a.succinct), ('exploration', a.exploration)]:
    subprocess.run([sys.executable, str(Path(__file__).with_name('run_native_qg.py')), 'measure',
                    '--zvec-source', str(source), '--bench-root', str(a.bench_root),
                    '--artifacts', str(a.artifacts / name)], check=True)
