#!/usr/bin/env python3
"""Resolve colocated r7i presets; no graph construction or performance claims."""
import argparse
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ann_benchmarks.algorithms.zvec_qg.module import digest, load_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=['r7i', 'r7i-exploration'], default='r7i')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    profiles = {}
    for ds, metric, dim in [('sift', 'euclidean', 128), ('gist', 'euclidean', 960), ('glove', 'angular', 100)]:
        c = load_profile(a.platform, metric, dim)
        for f in c['files']:
            if digest(f['path']) != f['sha256']:
                raise ValueError('Snapshot/provider fingerprint mismatch: ' + f['path'])
        profiles[ds] = c
    a.output.mkdir(parents=True, exist_ok=False)
    for ds, c in profiles.items():
        path = a.output / (ds + '.json')
        path.write_text(json.dumps(c, indent=2) + '\n')
        print(path)
    # A separate definitions root selects the requested variant without editing Git files.
    definition = yaml.safe_load((Path(__file__).resolve().parents[1] / 'ann_benchmarks/algorithms/zvec_qg/config.yml').read_text())
    definition['float']['any'][0]['base_args'][2] = str(a.output.resolve())
    dest = a.output / 'definitions/zvec_qg'
    dest.mkdir(parents=True)
    (dest / 'config.yml').write_text(yaml.safe_dump(definition, sort_keys=False))


if __name__ == '__main__':
    main()
