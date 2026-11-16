"""Summarize isolated full-dataset sweeps and paired optimization validation."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
from statistics import median

import h5py
import numpy as np


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def main(campaign):
    plan = json.loads((campaign / 'plan.json').read_text())
    records = []
    for file in sorted((campaign / 'measurements').rglob('measurements.jsonl')):
        records.extend(json.loads(line) for line in file.read_text().splitlines())
    root = Path(__file__).resolve().parents[1]
    truth = {}
    for dataset in {r['dataset'] for r in records}:
        with h5py.File(root / 'data' / (dataset + '.hdf5')) as f:
            truth[dataset] = (f['neighbors'][:, :10], f['distances'][:, 9] + .001)
    for r in records:
        gt, threshold = truth[r['dataset']]
        with h5py.File(r['result']) as f:
            ids, distances = f['neighbors'][:], f['distances'][:]
            assert f.attrs['best_search_time'] > 0 and np.all(f['times'][:] >= 0)
            assert np.isclose(r['qps'], 1 / f.attrs['best_search_time'], rtol=1e-12)
        assert ids.shape == gt.shape
        assert np.isfinite(r['qps']) and np.all(np.isfinite(distances))
        assert hashlib.sha256(ids.tobytes()).hexdigest() == r['ids_sha256']
        recall = int(np.count_nonzero(distances <= threshold[:, None])) / ids.size
        strict = sum(len(set(a) & set(b)) for a,b in zip(ids,gt)) / ids.size
        assert abs(recall-r['recall']) < 1e-12 and abs(strict-r['strict_id_recall']) < 1e-12
        r.update(recall=recall,strict_id_recall=strict)
    csv_write(campaign / 'all-measurements.csv', records)
    groups = {}
    for r in records:
        key = (r['profile'], r['build'], r['mode'], json.dumps(r['parameters'], sort_keys=True))
        if r['phase'] in ['paired', 'baseline']:
            groups.setdefault(key, []).append(r)
    aggregated = []
    for key, points in groups.items():
        assert len(points) == 2, (key, len(points))
        assert len({p['ids_sha256'] for p in points}) == 1, ('nondeterministic IDs', key)
        r = dict(points[0])
        r.update(qps=median(p['qps'] for p in points), qps_min=min(p['qps'] for p in points),
                 qps_max=max(p['qps'] for p in points), observations=len(points),
                 recall=min(p['recall'] for p in points), strict_id_recall=min(p['strict_id_recall'] for p in points))
        aggregated.append(r)
    csv_write(campaign / 'paired-and-baseline.csv', aggregated)
    gains, compiler_differences = [], []
    by_point = {}
    for r in aggregated:
        if r['phase'] == 'paired':
            by_point.setdefault((r['profile'], json.dumps(r['parameters'], sort_keys=True)), {})[(r['build'], r['mode'])] = r
    for key, points in by_point.items():
        legacy = points[('portable','legacy')]
        fast = points[('portable','native-list')]
        ice = points[('icelake','native-list')]
        for build in ['portable','icelake']:
            assert points[(build,'legacy')]['ids_sha256'] == points[(build,'native-list')]['ids_sha256'], key
        identical = len({p['ids_sha256'] for p in points.values()}) == 1
        if not identical:
            compiler_differences.append(dict(profile=key[0], parameters=fast['parameters'],
                portable_recall=fast['recall'], icelake_recall=ice['recall'],
                portable_strict=fast['strict_id_recall'], icelake_strict=ice['strict_id_recall']))
        gains.append(dict(dataset=fast['dataset'], variant=fast['variant'], profile=fast['profile'],
            parameters=fast['parameters'], recall=fast['recall'], strict_id_recall=fast['strict_id_recall'],
            legacy_qps=legacy['qps'], optimized_qps=fast['qps'], icelake_qps=ice['qps'],
            icelake_recall=ice['recall'], icelake_strict_id_recall=ice['strict_id_recall'],
            adapter_gain_pct=(fast['qps']/legacy['qps']-1)*100,
            icelake_change_pct=(ice['qps']/fast['qps']-1)*100, identical_ids_across_compilers=identical))
    csv_write(campaign / 'optimization-gains.csv', gains)
    comparisons = []
    for metric in ['recall', 'strict_id_recall']:
        for dataset in sorted({r['dataset'] for r in records}):
            for target in [.9,.95,.99]:
                for alg in ['exploration','succinct','qsgngt','descartes(01AI)']:
                    candidates = [r for r in aggregated if r['dataset']==dataset and r['variant']==alg
                                  and r[metric]>=target and (r['phase']=='baseline' or
                                      (r['build']=='portable' and r['mode']=='native-list'))]
                    if candidates:
                        r = max(candidates, key=lambda r:r['qps'])
                        comparisons.append(dict(selection_metric=metric, target=target, **r))
                    else:
                        comparisons.append(dict(selection_metric=metric, target=target, dataset=dataset,
                                                variant=alg, qps=None, recall=None, strict_id_recall=None))
    fields = sorted({k for row in comparisons for k in row})
    csv_write(campaign / 'comparison-90-95-99.csv', [{k:r.get(k) for k in fields} for r in comparisons])
    (campaign / 'comparison-90-95-99.json').write_text(json.dumps(comparisons, indent=2))
    coverage = []
    for job in plan['profiles']:
        rows = [r for r in records if r['profile']==job['id'] and r['phase']=='sweep']
        assert len(rows) == len(job['queries']), job['id']
        assert len({json.dumps(r['parameters'], sort_keys=True) for r in rows}) == len(rows)
        selected = json.loads((Path(job['directory']) / 'selected.json').read_text())
        paired = [r for r in records if r['profile']==job['id'] and r['phase']=='paired']
        assert len(paired) == len(selected)*8, job['id']
        coverage.append(dict(profile=job['id'], sweep_points=len(rows), paired_points=len(selected), paired_measurements=len(paired)))
    for job in plan['baselines']:
        rows = [r for r in records if r['profile']==job['id'] and r['phase']=='baseline']
        assert len(rows) == len(job['definition']['query_argument_groups'])*2, job['id']
    csv_write(campaign / 'coverage.csv', coverage)
    (campaign / 'validation.json').write_text(json.dumps(dict(
        sweep_points=sum(r['phase']=='sweep' for r in records), total_measurements=len(records),
        adapter_ids_identical=True, compiler_id_differences=compiler_differences,
        missing_threshold_cells=[{k:r[k] for k in ['dataset','variant','target','selection_metric']} for r in comparisons if r['qps'] is None],
        measurement='Median of two best-of-five observations for final comparisons; full query sets, CPU48, serial query timing after draining background benchmark containers.'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in sorted({r['dataset'] for r in records}):
        fig, ax = plt.subplots(figsize=(9,5))
        for variant in ['exploration','succinct','qsgngt','descartes(01AI)']:
            points = [r for r in records if r['dataset']==dataset and r['variant']==variant and r['phase'] in ['sweep','baseline']]
            frontier, best = [], 0
            for r in sorted(points, key=lambda r:(-r['recall'],-r['qps'])):
                if r['qps']>best:
                    frontier.append(r);best=r['qps']
            ax.plot([r['recall'] for r in frontier],[r['qps'] for r in frontier],'.-',label=variant)
        ax.set(xlabel='ANN Recall@10 (distance tolerance 0.001)',ylabel='QPS (best of five)',
               title=dataset,xlim=(.85,1.001),yscale='log')
        ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(campaign/(dataset+'.png'),dpi=160);plt.close(fig)
    print('Validated',len(records),'measurements,',len(gains),'paired configurations,',len(comparisons),'comparison cells')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('campaign',type=Path)
    main(p.parse_args().campaign)
