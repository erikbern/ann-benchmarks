#!/usr/bin/env python3
"""Isolated full-query ANN Python timings: three fresh processes, best-of-five median."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ann_benchmarks.algorithms.zvec_qg.module import ZvecQuantizedGraph, digest
from ann_benchmarks.runner import run_individual_query


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def guards(source):
    spec = importlib.util.spec_from_file_location('qg_protocol', source / 'tools/core/bench/qg_bench.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_quiet()
    require(os.environ.get('QG_ISOLATION_MANIFEST'), 'Use the isolate wrapper')
    require(os.sched_getaffinity(0) == {54}, 'Query CPU must be 54')
    require(psutil.Process().num_threads() == 1, 'Query process must be single-threaded')


def worker(a):
    guards(a.source)
    ds = a.dataset
    metric, dim = {'sift': ('euclidean', 128), 'gist': ('euclidean', 960), 'glove': ('angular', 100)}[ds]
    algo = ZvecQuantizedGraph(metric, dim, 'r7i', str(a.artifact))
    dataset = a.data / (algo.profile['dataset'] + '.hdf5')
    archive = ROOT / 'benchmark-configs/r7i-new-api/succinct/validation' / ds
    provider = json.loads((archive / 'provider.json').read_text())
    expected = next(f['sha256'] for f in provider['files'] if f['path'].endswith('.hdf5'))
    require(digest(dataset) == expected, 'Dataset fingerprint changed')
    with h5py.File(dataset) as f:
        train, queries = f['train'][:], f['test'][:]
        truth, thresholds = f['neighbors'][:, :10], f['distances'][:, 9]
    require(len(queries) == algo.profile['query_count'], 'Must use the full query set')
    algo.fit(train)
    native = json.loads((archive / 'measurements.json').read_text())
    dest = a.output / ds / ('repeat-' + str(a.repeat))
    dest.mkdir(parents=True, exist_ok=False)
    save(dest / 'runtime.json', dict(pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
        threads=psutil.Process().num_threads(), numa_maps=Path('/proc/self/numa_maps').read_text(),
        status=Path('/proc/self/status').read_text(), thp_policy=os.environ.get('QG_THP_POLICY'),
        python=sys.version, numpy=np.__version__, dataset_sha256=expected))
    rows = []
    for setting, point in enumerate(algo.profile['points']):
        algo.set_query_arguments(setting)
        guards(a.source)
        # Full runner warmup, including its outside-timer distance verification.
        run_individual_query(algo, train, queries, metric, 10, 1, False)
        reference = next(m['rounds'][0] for m in native if m['point']['name'] == point['name'] and m['repeat'] == 0)
        for round_index in range(5):
            guards(a.source)
            attrs, results = run_individual_query(algo, train, queries, metric, 10, 1, False)
            times = np.asarray([r[0] for r in results], dtype='<f8')
            ids = np.asarray([[item[0] for item in r[1]] for r in results], dtype='<u8')
            distances = np.asarray([[item[1] for item in r[1]] for r in results], dtype='<f8')
            require(ids.shape == distances.shape == (len(queries), 10), 'Top10 shape')
            require(np.isfinite(times).all() and (times > 0).all(), 'Invalid latency')
            require((ids < len(train)).all() and (np.diff(np.sort(ids, axis=1), axis=1) > 0).all(), 'Invalid IDs')
            ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
            require(ids_sha == reference['ids_sha256'], 'Results changed from validated native reference')
            exact = np.empty_like(distances)
            for start in range(0, len(queries), 256):
                x = train[ids[start:start + 256]].astype(np.float64)
                q = queries[start:start + len(x)].astype(np.float64)
                if metric == 'angular':
                    exact[start:start + len(x)] = 1 - np.einsum('nkd,nd->nk', x, q) / (np.linalg.norm(x, axis=2) * np.linalg.norm(q, axis=1)[:, None])
                else:
                    exact[start:start + len(x)] = np.sqrt(np.sum((x-q[:, None, :])**2, axis=2))
            require(np.allclose(distances, exact, rtol=3e-5, atol=3e-6), 'Independent distances differ')
            recall = float(np.mean(distances <= thresholds[:, None] + .001))
            independent_recall = float(np.mean(exact <= thresholds[:, None] + .001))
            strict = float(np.mean(np.any(ids[:, :, None] == truth[:, None, :], axis=2)))
            require(recall == independent_recall == reference['ann_recall'], 'Recall differs')
            require(strict == reference['strict_id_recall'] and recall >= point['target'], 'Strict recall or floor differs')
            raw = dest / (str(setting) + '-' + str(round_index) + '.hdf5')
            attrs.update(algo='zvec_qg', dataset=algo.profile['dataset'], repeat=a.repeat,
                         round_index=round_index, setting=setting, full_warmup=True,
                         ann_recall=recall, strict_id_recall=strict)
            with h5py.File(raw, 'w') as f:
                for key, value in attrs.items():
                    f.attrs[key] = value
                f.create_dataset('times', data=times)
                f.create_dataset('neighbors', data=ids)
                f.create_dataset('distances', data=distances)
            qps = 1 / float(times.mean())
            require(np.isclose(qps, 1 / attrs['best_search_time'], rtol=1e-12), 'QPS aggregation mismatch')
            rows.append(dict(dataset=ds, setting=setting, point=point, repeat=a.repeat,
                round=round_index, qps=qps, mean_query_us=float(times.mean())*1e6,
                ann_recall=recall, strict_id_recall=strict, query_count=len(queries),
                ids_sha256=ids_sha, max_exact_error=float(np.max(np.abs(distances-exact))),
                file=str(raw), sha256=digest(raw)))
            save(dest / 'measurements.json', rows)
        print('COMPLETE', ds, 'repeat', a.repeat, 'target', point['target'],
              'best5_qps', max(r['qps'] for r in rows[-5:]), flush=True)
    algo.done()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dataset', choices=['all', 'sift', 'gist', 'glove'], default='all')
    p.add_argument('--repeat', type=int, choices=[0, 1, 2])
    a = p.parse_args()
    if a.repeat is not None:
        require(a.dataset != 'all', 'Worker needs one dataset')
        return worker(a)
    guards(a.source)
    a.output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__).resolve(), ROOT/'ann_benchmarks/runner.py', ROOT/'ann_benchmarks/distance.py',
                  ROOT/'ann_benchmarks/algorithms/zvec_qg/module.py', ROOT/'ann_benchmarks/algorithms/zvec_qg/core.cpp']
    preset_paths = sorted((ROOT/'ann_benchmarks/algorithms/zvec_qg/configs/r7i').glob('*.json'))
    save(a.output/'run.json', dict(started_at=datetime.datetime.now().astimezone().isoformat(),
        ann_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in code_paths + preset_paths},
        artifact=json.loads(a.artifact.read_text()), argv=sys.argv,
        isolation_manifest=os.environ['QG_ISOLATION_MANIFEST'],
        protocol='full-query warmup; 3 fresh-process repetitions x 5 rounds; median of best5',
        timer='unmodified ANN run_individual_query: time.time around algo.query',
        cpu=54, smt_sibling=55, memory_node=0, index_rebuild_randomness_tested=False))
    summary = []
    for ds in (['sift', 'gist', 'glove'] if a.dataset == 'all' else [a.dataset]):
        all_rows = []
        for repeat in range(3):
            command = [sys.executable, str(Path(__file__).resolve()), '--artifact', str(a.artifact),
                       '--data', str(a.data), '--source', str(a.source), '--output', str(a.output),
                       '--dataset', ds, '--repeat', str(repeat)]
            with (a.output/(ds+'-'+str(repeat)+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            all_rows.extend(json.loads((a.output/ds/('repeat-'+str(repeat))/'measurements.json').read_text()))
        for setting in range(7):
            rows = [r for r in all_rows if r['setting'] == setting]
            require(len(rows) == 15 and len({r['ids_sha256'] for r in rows}) == 1, 'Missing/unstable rounds')
            reps = [max(r['qps'] for r in rows if r['repeat'] == i) for i in range(3)]
            qps = float(np.median(reps))
            summary.append(dict(dataset=ds, setting=setting, point=rows[0]['point'],
                ann_recall=rows[0]['ann_recall'], strict_id_recall=rows[0]['strict_id_recall'],
                repeat_best5_qps=reps, qps=qps, mean_query_us=1e6/qps,
                first_repeat_best5_qps=reps[0], repeat_spread=max(reps)/min(reps)-1))
        save(a.output/'summary.json', summary)
        print('DATASET COMPLETE', ds, flush=True)
    save(a.output/'complete.json', dict(completed_at=datetime.datetime.now().astimezone().isoformat(),
        settings=len(summary), measured_rounds=len(summary)*15,
        timed_queries=sum({'sift':10000, 'gist':1000, 'glove':10000}[r['dataset']]*15 for r in summary)))


if __name__ == '__main__':
    main()
