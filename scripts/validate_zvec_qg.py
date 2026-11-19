#!/usr/bin/env python3
"""Full-query correctness regression for the ANN bridge; produces NO performance ranking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ann_benchmarks.algorithms.zvec_qg.module import ZvecQuantizedGraph, digest


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--variant', choices=['succinct', 'exploration'], default='succinct')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    report = dict(purpose='correctness only; no formal QPS', full_query_passes=2,
                  variant=a.variant, artifact=json.loads(a.artifact.read_text()),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), measurements=rows,
                  passed=False, python=sys.version, numpy=np.__version__,
                  background_jobs_paused=False,
                  thp_policy=os.environ.get('QG_THP_POLICY', 'prepare'),
                  code_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
                      Path(__file__).resolve(),
                      ROOT / 'ann_benchmarks/algorithms/zvec_qg/module.py',
                      ROOT / 'ann_benchmarks/algorithms/zvec_qg/core.cpp']})
    platform = 'r7i' if a.variant == 'succinct' else 'r7i-exploration'
    for ds, metric, dim in [('sift', 'euclidean', 128), ('gist', 'euclidean', 960), ('glove', 'angular', 100)]:
        algo = ZvecQuantizedGraph(metric, dim, platform, str(a.artifact))
        with h5py.File(a.data / (algo.profile['dataset'] + '.hdf5')) as f:
            train, queries = f['train'][:], f['test'][:]
            truth, thresholds = f['neighbors'][:, :10], f['distances'][:, 9]
        algo.fit(train)
        archive = ROOT / 'benchmark-configs/r7i-new-api' / a.variant / 'validation' / ds
        preset = ROOT / 'ann_benchmarks/algorithms/zvec_qg/configs' / platform / (ds + '.json')
        native = json.loads((archive / 'measurements.json').read_text())
        for setting, point in enumerate(algo.profile['points']):
            algo.set_query_arguments(setting)
            ids = np.empty((len(queries), 10), dtype='<u8')
            scores = np.empty_like(ids, dtype='<f4')
            expanded = np.empty(len(queries), dtype='<u8')
            for i, q in enumerate(queries):
                ids[i] = algo.query(q, 10)
                d = algo.index.diagnostics()
                scores[i], expanded[i] = d['scores'][:10], d['expanded']
            require((ids < len(train)).all() and (np.diff(np.sort(ids, axis=1), axis=1) > 0).all(), 'ID range/uniqueness')
            require((expanded > 0).all(), 'Must execute QG')
            for i, q in enumerate(queries):
                require(np.array_equal(algo.query(q, 10), ids[i]), 'Unstable IDs')
                require(np.array_equal(algo.index.diagnostics()['scores'][:10], scores[i]), 'Unstable scores')
            exact = np.empty_like(scores, dtype=np.float64)
            for start in range(0, len(queries), 256):
                x = train[ids[start:start + 256]].astype(np.float64)
                q = queries[start:start + len(x)].astype(np.float64)
                if metric == 'angular':
                    exact[start:start + len(x)] = 1 - np.einsum('nkd,nd->nk', x, q) / (np.linalg.norm(x, axis=2) * np.linalg.norm(q, axis=1)[:, None])
                else:
                    exact[start:start + len(x)] = np.sum((x - q[:, None, :]) ** 2, axis=2)
            require(np.allclose(scores, exact, rtol=3e-5, atol=3e-6), 'Independent exact distance check')
            dist = exact if metric == 'angular' else np.sqrt(np.maximum(exact, 0))
            recall = float(np.mean(dist <= thresholds[:, None] + .001))
            strict = float(np.mean(np.any(ids[:, :, None] == truth[:, None, :], axis=2)))
            reference = next(m for m in native if m['point']['name'] == point['name'] and m['repeat'] == 0)['rounds'][0]
            ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
            scores_sha = hashlib.sha256(scores.tobytes()).hexdigest()
            require(ids_sha == reference['ids_sha256'], 'IDs differ from native entry')
            require(scores_sha == reference['scores_sha256'], 'Scores differ from native entry')
            require(recall == reference['ann_recall'] and strict == reference['strict_id_recall'], 'Recall differs')
            require(recall >= point['target'], 'Recall floor not reached')
            output = a.output / (ds + '-' + str(setting) + '.npz')
            np.savez_compressed(output, ids=ids, scores=scores, expanded=expanded)
            row = dict(dataset=ds, setting=setting, point=point['name'], target=point['target'],
                       query_count=len(queries), ann_recall=recall, strict_id_recall=strict,
                       ids_sha256=ids_sha, scores_sha256=scores_sha, native_bitwise_equal=True,
                       repeated_bitwise_equal=True, max_exact_error=float(np.max(np.abs(scores-exact))),
                       min_expanded=int(expanded.min()), raw=str(output), raw_sha256=digest(output))
            row['preset_sha256'] = digest(preset)
            rows.append(row)
            (a.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
            print(a.variant, ds, point['target'], recall, 'PASS', flush=True)
        # Boundary failures must be explicit, including the inherited BaseANN batch API.
        for q, k in [(np.full(dim, np.nan, dtype=np.float32), 10), (np.zeros(dim + 1, dtype=np.float32), 10), (queries[0], 11)]:
            try:
                algo.query(q, k)
            except ValueError:
                pass
            else:
                raise ValueError('Invalid query accepted')
        try:
            algo.batch_query(queries[:2], 10)
        except ValueError:
            pass
        else:
            raise ValueError('Concurrent batch accepted')
        algo.done()
        del train
    report['passed'] = True
    report['queries_checked'] = sum(r['query_count'] for r in rows) * 2
    (a.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
