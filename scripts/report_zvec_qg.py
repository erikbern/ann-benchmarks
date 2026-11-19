#!/usr/bin/env python3
"""Audit formal Python timings and compare with frozen, pre-campaign competitor records."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ann_benchmarks.algorithms.zvec_qg.module import digest


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--bench-root', required=True, type=Path)
    p.add_argument('--output', type=Path, help='Separate report directory; existing reports are preserved')
    a = p.parse_args()
    out = a.output or a.run / 'report'
    out.mkdir(exist_ok=False)
    summary = json.loads((a.run/'measurements/summary.json').read_text())
    complete = json.loads((a.run/'measurements/complete.json').read_text())
    require(complete['settings'] == len(summary) == 21, 'Incomplete campaign')
    rows = []
    for path in sorted((a.run/'measurements').glob('*/repeat-*/measurements.json')):
        rows.extend(json.loads(path.read_text()))
    require(len(rows) == 315, 'Missing measured rounds')
    for r in rows:
        require(digest(r['file']) == r['sha256'], 'Raw fingerprint')
        with h5py.File(r['file']) as f:
            times, ids = f['times'][:], f['neighbors'][:]
            require(np.isclose(1/times.mean(), r['qps'], rtol=1e-12), 'Raw QPS mismatch')
            require(hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest() == r['ids_sha256'], 'IDs fingerprint')
    for s in summary:
        candidates = [r for r in rows if r['dataset'] == s['dataset'] and r['setting'] == s['setting']]
        reps = [max(r['qps'] for r in candidates if r['repeat'] == i) for i in range(3)]
        require(reps == s['repeat_best5_qps'] and float(np.median(reps)) == s['qps'], 'Summary mismatch')
    run = json.loads((a.run/'measurements/run.json').read_text())
    cutoff = datetime.datetime.fromisoformat(run['started_at']).timestamp()
    competitors, rejected, coverage = [], [], []
    # Vendor revalidations live outside the standard result tree. Their JSONL
    # records provide dataset/algorithm provenance absent from some HDF5 attrs.
    extra = []
    base = a.bench_root/'results/comprehensive-icelake-20261116/measurements/baseline'
    for manifest in sorted(base.glob('*/*/measurements.jsonl')):
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            extra.append((Path(record['result']), record, str(manifest)))
    for ds, name in [('sift', 'sift-128-euclidean'), ('gist', 'gist-960-euclidean'), ('glove', 'glove-100-angular')]:
        with h5py.File(a.bench_root/'data'/(name+'.hdf5')) as f:
            truth = f['distances'][:, 9]
            strict_truth = f['neighbors'][:, :10]
            n = len(f['train'])
        for algo in ['qsgngt', 'descartes(01AI)']:
            valid = []
            paths = [(p, None, None) for p in sorted((a.bench_root/'results'/name/'10'/algo).glob('*.hdf5'))]
            paths += [(p, r, m) for p, r, m in extra if r['dataset'] == name and r['variant'] == algo]
            for path, provenance, manifest in paths:
                try:
                    # Interrupted/resumed background results must not enter this snapshot.
                    if path.stat().st_mtime >= cutoff:
                        continue
                    with h5py.File(path) as f:
                        if provenance is None:
                            require(f.attrs['algo'] == algo and f.attrs['dataset'] == name, 'Identity')
                        else:
                            require(json.loads(f.attrs['parameters_json']) == provenance['parameters'], 'Parameters/provenance')
                        require(f.attrs['count'] == 10 and f.attrs['run_count'] == 5 and not f.attrs.get('batch_mode', False), 'Protocol')
                        ids, dist, times = f['neighbors'][:], f['distances'][:], f['times'][:]
                        best = float(f.attrs['best_search_time'])
                        require(ids.shape == dist.shape == (len(truth), 10) and times.shape == (len(truth),), 'Shape')
                        require(np.isfinite(best) and best > 0 and np.isfinite(times).all() and (times > 0).all(), 'Timing')
                        live = (ids >= 0) & (ids < n)
                        require(((live & np.isfinite(dist)) | ((ids == -1) & np.isposinf(dist))).all(), 'IDs/distances')
                        sorted_ids = np.sort(ids, axis=1)
                        require(not ((sorted_ids[:,1:] == sorted_ids[:,:-1]) & (sorted_ids[:,1:] >= 0)).any(), 'Duplicates')
                        if provenance is not None:
                            require(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == provenance['ids_sha256'], 'Historical manifest ID hash')
                            require(np.isclose(1/best, provenance['qps'], rtol=1e-12), 'Historical manifest QPS')
                        valid.append(dict(dataset=ds, algorithm=algo, qps=1/best,
                            ann_recall=float(np.mean(dist <= truth[:,None]+.001)),
                            strict_id_recall=float(np.mean(np.any(ids[:,:,None] == strict_truth[:,None,:],axis=2))),
                            file=str(path), mtime=path.stat().st_mtime, provenance_manifest=manifest))
                except (OSError, KeyError, ValueError) as e:
                    rejected.append(dict(file=str(path), error=str(e)))
            coverage.append(dict(dataset=ds, algorithm=algo, valid_points=len(valid)))
            for target in [.3, .5, .7, .85, .9, .95, .99]:
                eligible = [r for r in valid if r['ann_recall'] >= target]
                winner = max(eligible, key=lambda r:r['qps']) if eligible else None
                if winner:
                    winner = dict(winner, sha256=digest(winner['file']))
                competitors.append(dict(dataset=ds, target=target, algorithm=algo, best=winner))
        # Validate the selected historical results against the same raw corpus;
        # this is offline correctness checking, not a performance rerun.
        selected = {r['best']['file'] for r in competitors if r['dataset'] == ds and r['best']}
        with h5py.File(a.bench_root/'data'/(name+'.hdf5')) as f:
            train, queries = f['train'][:], f['test'][:]
        for file in selected:
            with h5py.File(file) as f:
                ids, distances = f['neighbors'][:], f['distances'][:]
            exact = np.empty_like(distances, dtype=np.float64)
            for start in range(0, len(queries), 256):
                part = ids[start:start+256]
                x = train[np.maximum(part,0)].astype(np.float64)
                q = queries[start:start+len(part)].astype(np.float64)
                if ds == 'glove':
                    values = 1-np.einsum('nkd,nd->nk',x,q)/(np.linalg.norm(x,axis=2)*np.linalg.norm(q,axis=1)[:,None])
                else:
                    values = np.sqrt(np.sum((x-q[:,None,:])**2,axis=2))
                values[part < 0] = np.inf
                exact[start:start+len(part)] = values
            require(np.allclose(distances,exact,rtol=3e-5,atol=3e-6), 'Historical distance check: '+file)
            for record in competitors:
                if record['best'] and record['best']['file'] == file:
                    record['best']['independent_distance_validated'] = True
                    record['best']['independent_ann_recall'] = float(np.mean(exact <= truth[:,None]+.001))
        del train
    comparison = []
    lines = ['Zvec QG succinct: formal Python entry, median of 3 best-of-5 repetitions.',
             'Competitors: existing pre-campaign best-of-5 results; historical measurements, not rerun.',
             'Zvec first repetition best-of-5 is provided separately for matching aggregation.', '']
    for ds in ['sift', 'gist', 'glove']:
        lines += [ds.upper(), 'Floor | actual Zvec recall | Zvec median QPS | Zvec repeat-0 best5 | Huawei historical best5 | 01AI historical best5']
        for s in [r for r in summary if r['dataset'] == ds]:
            target=s['point']['target']
            rivals=[next(r['best'] for r in competitors if r['dataset']==ds and r['target']==target and r['algorithm']==algo) for algo in ['qsgngt','descartes(01AI)']]
            comparison.append(dict(dataset=ds,target=target,zvec=s,huawei=rivals[0],ai01=rivals[1]))
            values=[f"{r['qps']:,.0f}" if r else 'not reached' for r in rivals]
            lines.append(f"{target:.0%} | {s['ann_recall']:.5%} | {s['qps']:,.0f} | {s['first_repeat_best5_qps']:,.0f} | "+' | '.join(values))
        lines.append('')
    (out/'table.txt').write_text('\n'.join(lines).rstrip()+'\n')
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (out/'historical-snapshot.json').write_text(json.dumps(dict(cutoff=cutoff,coverage=coverage,rejected=rejected,winners=competitors),indent=2)+'\n')
    flat=[dict(dataset=s['dataset'],target=s['point']['target'],actual_recall=s['ann_recall'],strict_id_recall=s['strict_id_recall'],
               repeat0=s['repeat_best5_qps'][0],repeat1=s['repeat_best5_qps'][1],repeat2=s['repeat_best5_qps'][2],qps=s['qps'],mean_us=s['mean_query_us']) for s in summary]
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n");w.writeheader();w.writerows(flat)
    (out/'audit.json').write_text(json.dumps(dict(passed=True,raw_files=len(rows),timed_queries=sum(r['query_count'] for r in rows),
        all_round_qps_recomputed=True,all_raw_hashes_checked=True,competing_algorithms_rerun=False),indent=2)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
