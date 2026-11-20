"""Replay one frozen CPU0 benchmark process and check every ordered result."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['sift', 'gist', 'glove'])
    parser.add_argument('--data', type=Path, required=True, help='Original dataset HDF5 file')
    parser.add_argument('--output', type=Path, required=True, help='A new directory for this process')
    parser.add_argument('--repeat', type=int, choices=range(3), required=True)
    parser.add_argument('--preset', choices=['archived', 'recommended'],
                        default='archived', help='Select the archived settings or a CPU0 GloVe query preset')
    parser.add_argument('--provider', type=Path, help='Override the frozen provider path')
    parser.add_argument('--snapshot', type=Path, help='Override the GloVe QG snapshot path; its router must be adjacent')
    parser.add_argument('--runtime', type=Path, help='A relocated artifact manifest with the archived runtime hashes')
    args = parser.parse_args()
    if os.sched_getaffinity(0) != {0}:
        parser.error('Run with CPU affinity 0')
    folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(folder.parents[4]))
    from ann_benchmarks.runner import run_individual_query
    spec = importlib.util.spec_from_file_location(
        'ann_benchmarks.algorithms.zvec_qg.frozen_reference', folder / 'reference_adapter.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    manifest = read(folder / 'inputs.json')[args.dataset]
    saved = folder / args.dataset
    profile = read(saved / 'profile.json')
    if args.preset == 'archived':
        options = read(saved / 'selected.json')
        scores = [x for x in read(folder / 'scores.json') if x['dataset'] == args.dataset]
    else:
        if args.dataset != 'glove':
            parser.error('The recommended preset applies to GloVe')
        config = read(folder.parents[1] / 'configs/r7i-cpu0/glove.json')
        if config['dataset'] != profile['dataset']:
            raise ValueError('Query preset dataset differs from the frozen profile')
        scores = config[args.preset]
        options = [s['options'] for s in scores]
        profile['settings'] = [dict(target=s['target_recall'], options=s['options']) for s in scores]
    if ref.digest(args.data) != manifest['dataset_sha256']:
        raise ValueError('Dataset SHA-256 mismatch')
    if args.provider:
        profile['index_path'] = str(args.provider.resolve())
    if args.snapshot:
        if args.dataset != 'glove':
            parser.error('--snapshot applies to the GloVe sidecar')
        profile['storage']['local_qg_file'] = str(args.snapshot.resolve())
    if 'sidecar_sha256' in manifest:
        snap = Path(profile['storage']['local_qg_file'])
        if ref.digest(snap) != manifest['sidecar_sha256'] or ref.digest(str(snap) + '.router') != manifest['router_sha256']:
            raise ValueError('Snapshot or router SHA-256 mismatch')
    runtime = args.runtime or saved / 'artifact.json'
    expected = read(saved / 'artifact.json')
    actual = read(runtime)
    if (expected['runtime_sha256'][expected['extension']] != actual['runtime_sha256'][actual['extension']]
            or sorted(expected['runtime_sha256'].values()) != sorted(actual['runtime_sha256'].values())):
        raise ValueError('Runtime manifest does not identify the archived binaries')
    args.output.mkdir(parents=True, exist_ok=False)
    profile_file = args.output / 'profile.json'
    profile_file.write_text(json.dumps(profile, indent=2))
    with h5py.File(args.data, 'r') as data:
        train = data['train'][:]
        queries = data['test'][:]
        thresholds = data['distances'][:, 9]
    engine = ref.ZvecQuantizedGraph(profile['metric'], profile['dimension'], str(profile_file), str(runtime))
    engine.fit(train)
    rows = []
    try:
        settings = range(len(options))
        order = settings if args.repeat % 2 == 0 else reversed(settings)
        for setting in order:
            engine.set_query_arguments(options[setting])
            for query in queries:
                engine.query(query, 10)
            for iteration in range(5):
                _, result = run_individual_query(engine, train, queries, profile['metric'], 10, 1, False)
                ids = np.array([[i for i, _ in pairs] for _, pairs in result], dtype='<i8')
                distances = np.array([[d for _, d in pairs] for _, pairs in result])
                latency = np.array([dt for dt, _ in result])
                digest = hashlib.sha256(ids.tobytes()).hexdigest()
                recall = np.mean(distances <= thresholds[:, None] + .001)
                if digest != scores[setting]['ordered_ids_sha256'] or abs(recall - scores[setting]['actual_ann_recall']) > 1e-12:
                    raise ValueError('Ordered IDs or recall differ from the frozen run')
                if not np.all(np.isfinite(latency) & (latency > 0)):
                    raise ValueError('Invalid query latency')
                np.savez_compressed(args.output / f'{setting}-{iteration}.npz', ids=ids, distances=distances, latency=latency)
                rows.append(dict(setting=setting, iteration=iteration, qps=1 / latency.mean(), recall=recall, ids_sha256=digest))
                (args.output / 'rounds.json').write_text(json.dumps(rows, indent=2))
    finally:
        engine.done()


if __name__ == '__main__':
    main()
