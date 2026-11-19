#!/usr/bin/env python3
"""Check executable configs, committed measurements and selected presets.

Default: metadata-only review, without large external data/index files.
--raw: additionally verify every raw file hash and recompute per-round QPS.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def load(p):
    return json.loads(p.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def signature(p):
    return json.dumps({k: p[k] for k in ('profile', 'capacity', 'scan_limit', 'search')}, sort_keys=True)


def reject_legacy(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(not key.startswith('quantized_graph_') and key not in
                    ('use_quantized_graph', 'QuantizedGraph', 'QuantizedGraphRouter', 'ef_search'), 'legacy config field: ' + key)
            reject_legacy(child)
    elif isinstance(value, list):
        for child in value:
            reject_legacy(child)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--raw', action='store_true')
    a = ap.parse_args()
    seen_files = set()
    for ds in ('sift', 'gist', 'glove'):
        config = load(a.root / (ds + '.json'))
        best_config = load(a.root / ('best-' + ds + '.json'))
        reject_legacy(config)
        reject_legacy(best_config)
        proto = config['protocol']
        require((proto['topk'], proto['threads'], proto['warmup_rounds'], proto['rounds'], proto['repetitions']) ==
                (10, 1, 1, 5, 3), 'protocol mismatch')
        summary = load(a.root / 'validation' / ds / 'summary.json')
        measurements = load(a.root / 'validation' / ds / 'measurements.json')
        require(config['engine'] == best_config['engine'] == 'QuantizedGraphIndex', 'wrong engine')
        require(config['schema_version'] == 2 and summary['new_api_revalidated'], 'migration not validated')
        require(config['profiles'] == best_config['profiles'], 'winner changes build profile')
        for name, p in config['profiles'].items():
            require(p['encoding']['rotation_views'] == p['router']['rotation_views'], name + ': views mismatch')
            require(p['encoding']['codec'] in ('native', 'global', 'low_rank'), 'unknown codec')
            require(p['encoding']['codebook'] in ('hadamard', 'gaussian'), 'unknown codebook')
        require(len(measurements) == len(config['points']) * 3, 'missing repetitions')
        require(len(summary['points']) == len(config['points']), 'missing point summaries')
        unique = {}
        for point, row in zip(config['points'], summary['points']):
            require(point == row['point'], 'source point mismatch')
            require(row['full_query_count'] == config['query_count'], 'partial query set')
            entries = sorted([r for r in measurements if r['point']['name'] == point['name']], key=lambda r:r['repeat'])
            require([e['repeat'] for e in entries] == [0, 1, 2], 'repeat identities')
            hashes, scores, qps_each = set(), set(), []
            for entry in entries:
                require(entry['point'] == point and len(entry['rounds']) == 5, 'round count or config')
                qps_each.append(max(r['qps'] for r in entry['rounds']))
                require(entry['qps'] == qps_each[-1], 'repeat must use best of five')
                for r in entry['rounds']:
                    require(math.isfinite(r['qps']) and r['qps'] > 0 and r['min_qg_expanded'] > 0, 'QG/timer evidence')
                    require(r['ann_recall'] == row['ann_recall'] and r['strict_id_recall'] == row['strict_id_recall'], 'recall stability')
                    require(0 <= r['strict_id_recall'] <= r['ann_recall'] <= 1, 'recall ranges')
                    hashes.add(r['ids_sha256']); scores.add(r['scores_sha256'])
                    if a.raw:
                        for file in r['files']:
                            path = Path(file['path'])
                            if str(path) not in seen_files:
                                data = path.read_bytes()
                                require(len(data) == file['bytes'] and hashlib.sha256(data).hexdigest() == file['sha256'], 'raw fingerprint: ' + str(path))
                                seen_files.add(str(path))
                            if path.suffix == '.ns':
                                data = path.read_bytes()
                                ns = [v[0] for v in struct.iter_unpack('<Q', data)]
                                require(len(ns) == config['query_count'] and min(ns) > 0, 'raw latency shape')
                                require(math.isclose(1e9 / statistics.mean(ns), r['qps'], rel_tol=1e-12), 'round QPS recomputation')
            require(len(hashes) == len(scores) == 1, 'IDs/scores differ across repetitions')
            require(row['qps_each'] == qps_each and row['qps'] == statistics.median(qps_each), 'headline is not median of three')
            require(math.isclose(row['mean_query_us'], 1e6 / row['qps'], rel_tol=1e-12), 'mean query latency')
            unique.setdefault(signature(point), row)
        selected = set()
        for result in summary['winners']:
            eligible = [r for r in unique.values() if r['ann_recall'] >= result['target']]
            expected = max(eligible, key=lambda r:r['qps']) if eligible else None
            require(result['best'] == expected, 'winner is not the best eligible measured point')
            if expected:
                selected.add(expected['point']['name'])
        require({p['name'] for p in best_config['points']} == selected, 'executable best config differs from winners')
        for p in best_config['points']:
            require(p in config['points'], 'unmeasured query parameters in best config')
        print(ds, len(config['points']), 'points x 3 repeats x 5 rounds validated')
    print('PASS; raw files checked:', len(seen_files))


if __name__ == '__main__':
    main()
