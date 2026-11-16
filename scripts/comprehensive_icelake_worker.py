"""Build immutable snapshots and measure full public query sets for the suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import yaml


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def large_mappings():
    """Per-VMA evidence; process-wide AnonHugePages is not index coverage."""
    regions = []
    current = None
    for line in Path('/proc/self/smaps').read_text().splitlines():
        fields = line.split()
        if fields and '-' in fields[0] and ':' not in fields[0]:
            current = {'mapping': line}
            regions.append(current)
        elif current is not None and fields and fields[0] in ('Size:', 'Rss:', 'AnonHugePages:', 'THPeligible:'):
            current[fields[0][:-1]] = int(fields[1])
    return [r for r in regions if r.get('Size', 0) >= 65536]


def pin_query_threads(cpu):
    # Some vendor libraries retain threads created during fit(). Pin those too;
    # newly created threads inherit the calling thread's single-CPU affinity.
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), {cpu})
        except ProcessLookupError:
            pass


def convert(dataset, output, converter):
    """Use official serialized metadata; stream exact FP32 bytes and row IDs.

    VecsHeader layout is tools/core/vecs_common.h (pack(4)). Validate its
    metadata, offsets, IDs and payload against official txt2vecs first.
    """
    output = Path(output)
    if output.exists():
        return
    header = struct.Struct('<QHHI11Q')
    with h5py.File(dataset) as f:
        data = f['train']
        n, dim = data.shape
        assert data.dtype == np.float32
        sample = data[:4]
        text = output.with_suffix('.sample.txt')
        text.write_text(''.join(str(i) + ';' + ' '.join(format(float(v), '.9g') for v in row) + '\n'
                                for i, row in enumerate(sample)))
        fixture = output.with_suffix('.sample.vecs')
        subprocess.run([str(converter), '--input=' + str(text), '--output=' + str(fixture),
                        '--dimension=' + str(dim), '--method=L2'], check=True)
        content = fixture.read_bytes()
        h = list(header.unpack_from(content))
        assert header.size == 104 and h[:3] == [4, 0, 1] and h[4] == 3
        offset = header.size + h[3]
        assert h[5:9] == [4 * dim * 4, 32, 0, 4 * dim * 4], h
        assert content[offset:] == sample.astype('<f4').tobytes() + np.arange(4, dtype='<u8').tobytes()
        h[0], h[5], h[6], h[8] = n, n * dim * 4, n * 8, n * dim * 4
        tmp = output.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            stream.write(header.pack(*h))
            stream.write(content[header.size:offset])
            for start in range(0, n, 8192):
                np.asarray(data[start:start + 8192], dtype='<f4').tofile(stream)
            np.arange(n, dtype='<u8').tofile(stream)
        assert tmp.stat().st_size == offset + n * (8 + dim * 4)
        # Validate random rows at their final file offsets, not just a fixture.
        with tmp.open('rb') as stream:
            for i in [0, n // 3, n // 2, n - 1]:
                stream.seek(offset + n * dim * 4 + i * 8)
                assert struct.unpack('<Q', stream.read(8))[0] == i
                stream.seek(offset + i * dim * 4)
                np.testing.assert_array_equal(np.frombuffer(stream.read(dim * 4), dtype='<f4'), data[i])
        tmp.replace(output)
        save_json(output.with_suffix('.validation.json'), dict(rows=n, dimension=dim,
                  exact_fp32=True, official_fixture_payload_match=True, sampled_full_file_rows=4))


def build(job):
    source = Path(job['source'])
    root = Path(job['directory'])
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'index'
    if (root / 'ready.json').exists():
        assert target.exists()
        return
    os.sched_setaffinity(0, set(job['cpus']))
    short = job['dataset'].split('-')[0]
    configs = source / 'tools/core/bench/config'
    if short == 'sift':
        c = yaml.safe_load((configs / f"sift_fp32_v{job['views']}_build.yaml").read_text())
        c['BuilderCommon']['InputIndex'] = job['raw_index']
    else:
        c = yaml.safe_load((configs / f'{short}_fp32_build.yaml').read_text())
        c['BuilderCommon'].update(BuildFile=job['vectors'], IndexPath=str(root / 'work.index'))
    c['BuilderCommon']['DumpPath'] = str(target)
    c['QuantizedGraph']['RotationViews'] = job['views']
    path = root / 'build.yaml'
    path.write_text(yaml.safe_dump(c))
    start = time.time()
    subprocess.run([str(source / 'build-ann/bin/local_builder'), str(path)], check=True)
    save_json(root / 'ready.json', dict(build_seconds=time.time()-start, index_bytes=target.stat().st_size,
                                      dataset=job['dataset'], config=c))


def load_data(dataset):
    with h5py.File(ROOT / 'data' / (dataset + '.hdf5')) as f:
        return f['train'][:], f['test'][:], f['neighbors'][:, :10], f['distances'][:, 9] + .001, str(f.attrs['distance'])


def measure(algo, data, params, folder, metadata, runs=5):
    from ann_benchmarks.runner import run_individual_query
    saved = folder / 'measurements.jsonl'
    if saved.exists():
        completed = [json.loads(line) for line in saved.read_text().splitlines()]
        matches = [r for r in completed if r['parameters'] == params]
        assert len(matches) <= 1, (folder, params)
        if matches:
            with h5py.File(matches[0]['result']) as f:
                assert f['neighbors'].shape == data[2].shape
                assert f.attrs['best_search_time'] > 0
            return matches[0]
    train, queries, truth, threshold, metric = data
    attrs, answers = run_individual_query(algo, train, queries, metric, 10, runs, False)
    ids = np.asarray([[i for i, _ in row] for _, row in answers], dtype=np.int64)
    distances = np.asarray([[d for _, d in row] for _, row in answers])
    assert ids.shape == truth.shape and np.all((ids >= 0) & (ids < len(train)))
    # Aggregate integer hits first so an exact 99% is not rounded just below
    # the target by repeated floating-point division of individual queries.
    recall = int(np.count_nonzero(distances <= threshold[:, None])) / ids.size
    strict = sum(len(set(a) & set(b)) for a, b in zip(ids, truth)) / ids.size
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (key + '.hdf5')
    with h5py.File(target, 'w') as f:
        f.create_dataset('neighbors', data=ids, compression='gzip')
        f.create_dataset('distances', data=distances, compression='gzip')
        f.create_dataset('times', data=[t for t, _ in answers])
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['parameters_json'] = json.dumps(params, sort_keys=True)
    record = dict(**metadata, parameters=params, recall=recall, strict_id_recall=strict,
                  qps=1/attrs['best_search_time'], query_count=len(queries),
                  ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), result=str(target), time=time.time())
    with (folder / 'measurements.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print('MEASURED', json.dumps(record), flush=True)
    return record


def query(job, native_build, phase, cycle):
    from py4bench import CoreSearcher
    from ann_benchmarks.algorithms.base.module import BaseANN
    import zvec
    expected_overlay = Path(job['campaign']) / 'overlays' / f"{job['variant']}-{native_build}"
    assert Path(zvec.__file__).is_relative_to(expected_overlay), zvec.__file__
    pin_query_threads(job['query_cpu'])
    data = load_data(job['dataset'])
    source = Path(job['source'])
    short = job['dataset'].split('-')[0]
    c = yaml.safe_load((source / f'tools/core/bench/config/{short}_fp32_recall_bench.yaml').read_text())
    index_params = json.loads(c['IndexCommon']['IndexConfig'])
    params = job['queries'] if phase == 'sweep' else json.loads((Path(job['directory']) / 'selected.json').read_text())
    rows = []
    before_load = large_mappings()
    for number, param in enumerate(params):
        searcher = CoreSearcher(str(Path(job['directory']) / 'index'), index_params, param)
        if number == 0:
            save_json(Path(job['directory']) / f'mappings-{phase}-{native_build}-{cycle}.json',
                      dict(module=zvec.__file__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                           before_index_load=before_load, after_index_load=large_mappings(),
                           note='Large VMAs before/after load, sizes in KiB. Do not attribute process-wide totals to the graph.'))
        modes = ['native-list'] if phase == 'sweep' else ['legacy', 'native-list']
        if cycle % 2:
            modes.reverse()
        hashes = set()
        for mode in modes:
            folder = Path(job['campaign']) / 'measurements' / phase / job['id'] / native_build / str(cycle) / mode
            algo = BaseANN()
            algo.name = f"{job['id']}({native_build},{mode},{param})"
            algo.query = searcher.search_list if mode == 'native-list' else (
                lambda q, k: searcher.search(np.require(q, dtype=np.float32, requirements=['C', 'A']), k))
            r = measure(algo, data, param, folder, dict(dataset=job['dataset'], variant=job['variant'],
                        profile=job['id'], build=native_build, phase=phase, cycle=cycle, mode=mode))
            rows.append(r)
            hashes.add(r['ids_sha256'])
        assert len(hashes) == 1, (job['id'], param, hashes)
        del algo, searcher
    if phase == 'sweep':
        selected = []
        for metric in ['recall', 'strict_id_recall']:
            for target in [.9, .95, .99]:
                eligible = [r for r in rows if r[metric] >= target]
                if eligible:
                    best = max(eligible, key=lambda r:r['qps'])['parameters']
                    if best not in selected:
                        selected.append(best)
        assert selected, job['id']
        save_json(Path(job['directory']) / 'selected.json', selected)


def baseline(job):
    from ann_benchmarks.definitions import Definition, instantiate_algorithm
    os.sched_setaffinity(0, set(job['cpus']))
    os.chdir(job['work'])
    data = load_data(job['dataset'])
    algo = instantiate_algorithm(Definition(**job['definition']))
    start = time.time()
    algo.fit(data[0])
    save_json(Path(job['directory']) / 'ready.json', dict(build_seconds=time.time()-start))
    release = Path(job['directory']) / 'query'
    while not release.exists():
        time.sleep(2)
    pin_query_threads(job['query_cpu'])
    for cycle in range(2):
        for param in job['definition']['query_argument_groups']:
            algo.set_query_arguments(*param)
            folder = Path(job['campaign']) / 'measurements/baseline' / job['id'] / str(cycle)
            measure(algo, data, param, folder, dict(dataset=job['dataset'], variant=job['definition']['algorithm'],
                    profile=job['id'], build='vendor', phase='baseline', cycle=cycle, mode='native'))
    algo.done()
    (Path(job['directory']) / 'done').touch()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path)
    p.add_argument('--action', choices=['build', 'query', 'baseline', 'convert'], required=True)
    p.add_argument('--native-build', default='portable')
    p.add_argument('--phase', default='sweep')
    p.add_argument('--cycle', type=int, default=0)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--converter', type=Path)
    a = p.parse_args()
    if a.action == 'convert':
        convert(a.dataset, a.output, a.converter)
    else:
        job = json.loads(a.job.read_text())
        if a.action == 'build':
            build(job)
        elif a.action == 'baseline':
            baseline(job)
        else:
            query(job, a.native_build, a.phase, a.cycle)
