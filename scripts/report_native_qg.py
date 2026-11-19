#!/usr/bin/env python3
"""Summarize complete native runs; keep historical timing and ID ties explicit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def distances(train, queries, ids, cosine):
    result = np.empty(ids.shape, dtype=np.float64)
    for start in range(0, len(queries), 256):
        x = train[ids[start:start + 256]].astype(np.float64)
        q = queries[start:start + len(x)].astype(np.float64)
        if cosine:
            d = 1 - np.einsum('nkd,nd->nk', x, q) / (np.linalg.norm(x, axis=2) * np.linalg.norm(q, axis=1)[:, None])
        else:
            d = np.sqrt(np.sum((x - q[:, None, :]) ** 2, axis=2))
        result[start:start + len(x)] = d
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', required=True, type=Path)
    p.add_argument('--bench-root', required=True, type=Path)
    p.add_argument('--history', required=True, type=Path)
    a = p.parse_args()
    out = a.artifacts / 'report'
    out.mkdir(exist_ok=True)
    rows, paired, capacity = [], [], []
    for ds in ('sift', 'gist', 'glove'):
        history = load(a.history / (ds + '.json'))
        with h5py.File(a.bench_root / 'data' / (history['dataset'] + '.hdf5'), 'r') as f:
            train, query = f['train'][:], f['test'][:]
        for variant in ('succinct', 'exploration'):
            run = a.artifacts / variant / ds
            c, summary = load(run / 'resolved.json'), load(run / 'summary.json')
            assert summary['new_api_revalidated'] and len(summary['points']) == 21
            points = {r['point']['name']: r for r in summary['points']}
            for i, old in enumerate(history['presets']):
                target = old['target_ann_recall']
                winner = next(r['best'] for r in summary['winners'] if r['target'] == target)
                migrated = points['migration-' + str(i)]
                row = dict(dataset=ds, variant=variant, target=target,
                           old_qps_median_2=old['revalidation']['qps_median'],
                           old_actual_recall=old['revalidation']['ann_recall'],
                           migrated_qps=migrated['qps'], migrated_actual_recall=migrated['ann_recall'],
                           new_qps=winner['qps'] if winner else None,
                           new_actual_recall=winner['ann_recall'] if winner else None,
                           new_strict_id_recall=winner['strict_id_recall'] if winner else None,
                           qps_each=winner['qps_each'] if winner else [],
                           mean_query_us=winner['mean_query_us'] if winner else None,
                           winner=winner['point'] if winner else None,
                           source_commit=summary['source']['commit'],
                           timing_directly_comparable_to_history=False)
                rows.append(row)
                oldfile = a.bench_root / old['revalidation']['results'][0]
                with h5py.File(oldfile, 'r') as f:
                    oldids = f['neighbors'][:]
                    olddist = f['distances'][:]
                profile = migrated['point']['profile']
                prefix = run / 'raw/migration/0' / profile / f'migration-{i}-round0'
                newids = np.fromfile(str(prefix) + '.ids', dtype='<u8').reshape(len(query), 10)
                newdist = distances(train, query, newids, ds == 'glove')
                old_exact = distances(train, query, oldids, ds == 'glove')
                ordered = np.all(oldids == newids, axis=1)
                sets_equal = np.all(np.sort(oldids, axis=1) == np.sort(newids, axis=1), axis=1)
                distance_equal = np.all(np.abs(np.sort(old_exact, axis=1) - np.sort(newdist, axis=1)) <= 1e-6, axis=1)
                new_returned = np.fromfile(str(prefix) + '.scores', dtype='<f4').reshape(len(query), 10)
                if ds != 'glove':
                    new_returned = np.sqrt(np.maximum(new_returned, 0))
                qi, oi, ni = np.where(oldids[:, :, None] == newids[:, None, :])
                differing = [dict(query_row=int(q), old_ids=oldids[q].tolist(), new_ids=newids[q].tolist(),
                                  old_independent_distances=old_exact[q].tolist(),
                                  new_independent_distances=newdist[q].tolist(),
                                  equal_distances_within_tolerance=bool(distance_equal[q]))
                             for q in np.flatnonzero(~ordered)]
                paired.append(dict(dataset=ds, variant=variant, target=target, query_count=len(query),
                    old_source=fingerprint(oldfile), new_ids=fingerprint(str(prefix) + '.ids'),
                    ordered_ids_identical_queries=int(ordered.sum()), same_id_set_queries=int(sets_equal.sum()),
                    different_ids_equal_distances_queries=int((~sets_equal & distance_equal).sum()),
                    different_ids_different_distances_queries=int((~sets_equal & ~distance_equal).sum()),
                    reordered_same_ids_queries=int((sets_equal & ~ordered).sum()),
                    tie_distance_absolute_tolerance=1e-6,
                    old_framework_distance_max_error=float(np.max(np.abs(olddist - old_exact))),
                    distance_sources='old: ANN-Benchmarks recomputed HDF5 distances; new: native exact callback scores (sqrt for L2). Old native callback scores were not retained.',
                    common_ids_max_distance_difference=float(np.max(np.abs(olddist[qi, oi] - new_returned[qi, ni]))) if len(qi) else None,
                    differing_queries=differing,
                    new_returned_distances_independently_validated=True))
                alternate = points['capacity-' + str(i)]
                capacity.append(dict(dataset=ds, variant=variant, target=target,
                    old_ef_seeded_capacity=migrated['point']['capacity'], alternate_capacity=alternate['point']['capacity'],
                    full_query_ids_equal=migrated['ids_sha256'] == alternate['ids_sha256'],
                    seeded_recall=migrated['ann_recall'], alternate_recall=alternate['ann_recall'],
                    seeded_qps=migrated['qps'], alternate_qps=alternate['qps'],
                    note='Controlled new API capacity perturbation; not proof of universal ef_search equivalence.'))
    for name, value in [('comparison.json', rows), ('paired-correctness.json', paired), ('capacity-validation.json', capacity)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    with (out / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['Full-query native QG results: median of three best-of-five repetitions.',
             'Historical Python-call QPS is NOT directly comparable; no kernel-speedup claim.',
             'Each variant uses its own freshly built QG snapshots and the same verified base graph.', '']
    for ds in ('sift', 'gist', 'glove'):
        lines += [ds.upper(), 'Recall floor | succinct QPS / actual recall | exploration QPS / actual recall']
        for target in (.3, .5, .7, .85, .9, .95, .99):
            values = []
            for variant in ('succinct', 'exploration'):
                r = next(r for r in rows if r['dataset'] == ds and r['variant'] == variant and r['target'] == target)
                values.append(f"{r['new_qps']:,.0f} / {r['new_actual_recall']:.5%}" if r['new_qps'] is not None else 'not reached')
            lines.append(f'{target:.0%} | ' + ' | '.join(values))
        lines.append('')
    (out / 'table.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
