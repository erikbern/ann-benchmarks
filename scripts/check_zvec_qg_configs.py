#!/usr/bin/env python3
"""Check the colocated presets against immutable native measurements (no data needed)."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def no_legacy(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(not key.startswith('quantized_graph_') and key not in
                    ('ef_search', 'use_quantized_graph', 'QuantizedGraph', 'QuantizedGraphRouter'), 'Legacy key: ' + key)
            no_legacy(child)
    elif isinstance(value, list):
        for child in value:
            no_legacy(child)


def main():
    for variant, platform in [('succinct', 'r7i'), ('exploration', 'r7i-exploration')]:
        archive = ROOT / 'benchmark-configs/r7i-new-api' / variant
        for ds in ['sift', 'gist', 'glove']:
            c = json.loads((ROOT / 'ann_benchmarks/algorithms/zvec_qg/configs' / platform / (ds + '.json')).read_text())
            orig = json.loads((archive / ('best-' + ds + '.json')).read_text())
            no_legacy(c)
            require(c['variant'] == variant and c['engine'] == 'QuantizedGraphIndex', 'Variant/engine')
            require(c['train_shape'] == [orig['train_count'], orig['dimension']], 'Shape')
            require([p['target'] for p in c['points']] == [.3, .5, .7, .85, .9, .95, .99], 'Seven recall floors')
            for point, (target, name) in zip(c['points'], orig['recommended_points'].items()):
                expected = copy.deepcopy(next(p for p in orig['points'] if p['name'] == name))
                expected['target'] = float(target)
                require(point == expected, 'Query parameter drift: ' + ds)
            manifests = [json.loads((archive / 'validation' / ds / 'provider.json').read_text())]
            for name, profile in c['profiles'].items():
                require(profile['encoding'] == orig['profiles'][name]['encoding'], 'Encoding drift')
                require(profile['router'] == orig['profiles'][name]['router'], 'Router drift')
                require(profile['encoding']['rotation_views'] == profile['router']['rotation_views'], 'Rotation views')
                require(profile['index_prefix'] == '${ZVEC_QG_INDEX_ROOT}/' + ds + '/' + name, 'Index path')
                manifests.append(json.loads((archive / 'validation' / ds / ('index-' + name + '.json')).read_text()))
            expected = {Path(f['path']).name: (f['sha256'], f['bytes']) for m in manifests for f in m['files']
                        if Path(f['path']).name.startswith('provider.') or Path(f['path']).suffix in ('.qg', '.router')}
            actual = {Path(f['path']).name: (f['sha256'], f['bytes']) for f in c['files']}
            require(actual == expected, 'File fingerprint drift')
            print(platform, ds, 'PASS: seven archived winners and exact artifact fingerprints')


if __name__ == '__main__':
    main()
