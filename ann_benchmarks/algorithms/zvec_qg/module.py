"""Independent QG API, owned CSR provider, fingerprinted load-only ANN adapter."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from ..base.module import BaseANN


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_config(path):
    return json.loads(Path(os.path.expandvars(str(path))).read_text())


def load_profile(profile, metric, dimension):
    preset_dir = Path(__file__).parent / 'configs' / profile if profile in ('r7i', 'r7i-exploration') else Path(os.path.expandvars(profile))
    if preset_dir.is_dir():
        profiles = [read_config(p) for p in preset_dir.glob('*.json')]
        matches = [p for p in profiles if p['ann_metric'] == metric and p['dimension'] == dimension]
        if len(matches) != 1:
            raise ValueError('No unique platform preset for metric/dimension')
        config = matches[0]
    else:
        config = read_config(profile)
    if config['ann_metric'] != metric or config['dimension'] != dimension:
        raise ValueError('Dataset metric/dimension mismatch')
    def reject_legacy(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.startswith('quantized_graph_') or key in ('use_quantized_graph', 'ef_search', 'QuantizedGraph', 'QuantizedGraphRouter'):
                    raise ValueError('Legacy QG configuration key: ' + key)
                reject_legacy(child)
        elif isinstance(value, list):
            for child in value:
                reject_legacy(child)
    reject_legacy(config)
    if config['engine'] != 'QuantizedGraphIndex' or config['precision'] != 'FP32':
        raise ValueError('Only independent QG FP32 presets are supported')
    for p in config['profiles'].values():
        if p['encoding']['rotation_views'] != p['router']['rotation_views']:
            raise ValueError('Encoding/router rotation views differ')
    for point in config['points']:
        options = point['search']
        if (type(point['capacity']) is not int or point['capacity'] <= 0 or
                type(point['scan_limit']) is not int or point['scan_limit'] < 0 or
                point['profile'] not in config['profiles'] or
                set(options) != {'margin', 'bucket_width', 'max_expansions', 'route_beam', 'uncertainty_bonus'}):
            raise ValueError('Invalid native query options')
        if (not all(np.isfinite(v) and v >= 0 for v in options.values()) or
                any(type(options[k]) is not int or options[k] >= 2**32 for k in ('max_expansions', 'route_beam'))):
            raise ValueError('Native query options must be finite and in range')
    def resolve(value):
        if isinstance(value, str) and '${' in value:
            value = os.path.expandvars(value)
            if '${' in value:
                raise ValueError('Set environment variables for ' + value)
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(config)


class ZvecQuantizedGraph(BaseANN):
    def __init__(self, metric, dimension, profile, artifact):
        self.profile = load_profile(profile, metric, dimension)
        self.artifact = read_config(artifact)
        if self.profile['variant'] != self.artifact['variant']:
            raise ValueError('Source variant differs from the preset')
        for path, sha in self.artifact['runtime_sha256'].items():
            if digest(path) != sha:
                raise ValueError('Runtime fingerprint mismatch: ' + path)
        if digest(Path(__file__).with_name('core.cpp')) != self.artifact['bridge_sha256']:
            raise ValueError('Rebuild the bridge after modifying core.cpp')
        spec = importlib.util.spec_from_file_location('_zvec_qg_core', self.artifact['extension'])
        self.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.core)
        self.index = None
        self.name = 'zvec_qg ' + self.profile['variant'] + ' ' + self.profile['dataset']

    def fit(self, X):
        if self.index is not None:
            raise RuntimeError('Close the index before loading another')
        if X.dtype != np.float32 or not X.flags.c_contiguous or list(X.shape) != self.profile['train_shape']:
            raise ValueError('Expected full contiguous FP32 training corpus')
        if hashlib.sha256(memoryview(X)).hexdigest() != self.profile['train_sha256']:
            raise ValueError('Training corpus fingerprint mismatch')
        for item in self.profile['files']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('Provider/snapshot fingerprint mismatch: ' + item['path'])
        self.index = self.core.Index(json.dumps(self.profile))
        # Load the first snapshot inside fit so the ANN memory/load accounting includes QG.
        self.set_query_arguments(0)

    def set_query_arguments(self, setting):
        if type(setting) is not int or not 0 <= setting < len(self.profile['points']):
            raise ValueError('Expected a valid frozen query setting index')
        if self.index is None:
            raise RuntimeError('Load the index before configuring queries')
        self.index.configure(json.dumps(self.profile['points'][setting]))
        self.setting = setting

    def query(self, q, n):
        if self.index is None:
            raise RuntimeError('Load and configure the index before querying')
        q = np.asarray(q, dtype=np.float32, order='C')
        if q.shape != (self.profile['dimension'],) or not np.isfinite(q).all() or n != 10:
            raise ValueError('Expected finite query vector and top10')
        if self.profile['metric'] == 'cosine':
            normalized = q.astype(np.float64)
            # Match native campaign preprocessing: numpy.linalg.norm(..., axis=1).
            norm = np.sqrt(np.sum(normalized * normalized))
            if norm == 0:
                raise ValueError('Cosine query must have nonzero norm')
            q = np.asarray(normalized / norm, dtype=np.float32)
        return self.index.search(q, n)

    def batch_query(self, X, n):
        raise ValueError('Only single-query, single-thread measurements are supported')

    def get_additional(self):
        return dict(source_commit=self.artifact['source_commit'], entry='QuantizedGraphIndex',
                    query_threads=1, borrowed_corpus_bytes=0, variant=self.profile['variant'],
                    active_profile=self.profile['points'][self.setting]['profile'],
                    process_rss_bytes=int(self.get_memory_usage() * 1024),
                    fit_is_load_only=True, query_timer='Python + normalization + QG + result array',
                    native_campaign_qps_comparable=False)

    def done(self):
        self.index = None
