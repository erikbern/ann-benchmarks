"""Single-query ANN adapter for the r7i succinct snapshots.

ZVEC_QG_ARTIFACT points to the matching native extension manifest.
ZVEC_QG_BENCH_ROOT can point to the directory containing data/ and results/.
"""
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
    config = resolve(config)
    root = Path(os.environ.get('ZVEC_QG_BENCH_ROOT', '.'))
    config['provider_prefix'] = str(root / config['provider_prefix'])
    for build in config['profiles'].values():
        build['index_prefix'] = str(root / build['index_prefix'])
    for item in config['files']:
        item['path'] = str(root / item['path'])
    return config


class ZvecQuantizedGraph(BaseANN):
    def __init__(self, metric, dimension, profile, artifact):
        preset = (Path(__file__).parent / 'configs/r7i-native-builder'
                  if profile == 'r7i' else Path(os.path.expandvars(profile)))
        if preset.is_dir():
            paths = []
            for path in sorted(preset.glob('*.json')):
                header = read_config(path)
                if (header['ann_metric'], header['dimension']) == (metric, dimension):
                    paths.append(path)
        else:
            paths = [preset]
        if not paths:
            raise ValueError('No r7i QG preset for this metric and dimension')
        self.configs = [load_profile(path, metric, dimension) for path in paths]
        self.profile = self.configs[0]
        self.settings = sorted(
            [(config, point) for config in self.configs for point in config['points']],
            key=lambda pair: pair[1]['target'])
        self.artifact = read_config(artifact)
        for config in self.configs:
            if config['variant'] != self.artifact['variant']:
                raise ValueError('Source variant differs from the preset')
            if config.get('source_commit', self.artifact['source_commit']) != self.artifact['source_commit']:
                raise ValueError('Runtime source commit differs from the preset')
        if self.artifact['extension'] not in self.artifact['runtime_sha256']:
            raise ValueError('Missing QG extension fingerprint')
        for path, sha in self.artifact['runtime_sha256'].items():
            if digest(path) != sha:
                raise ValueError('Runtime fingerprint mismatch: ' + path)
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
        corpus_sha = hashlib.sha256(memoryview(X)).hexdigest()
        for config in self.configs:
            if corpus_sha != config['train_sha256'] or list(X.shape) != config['train_shape']:
                raise ValueError('Training corpus fingerprint mismatch')
            for item in config['files']:
                if digest(item['path']) != item['sha256']:
                    raise ValueError('Provider/snapshot fingerprint mismatch: ' + item['path'])
        self._configure(0)

    def set_query_arguments(self, setting):
        if type(setting) is not int or not 0 <= setting < len(self.settings):
            raise ValueError('Expected a valid frozen query setting index')
        if self.index is None:
            raise RuntimeError('Load the index before configuring queries')
        self._configure(setting)

    def _configure(self, setting):
        config, point = self.settings[setting]
        if self.index is None or self.profile is not config:
            # GloVe N32 and N64 have different CSR graphs as well as snapshots.
            self.index = self.core.Index(json.dumps(config))
            self.profile = config
        self.index.configure(json.dumps(point))
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
                    active_profile=self.settings[self.setting][1]['profile'],
                    process_rss_bytes=int(self.get_memory_usage() * 1024),
                    fit_is_load_only=True, query_timer='Python + normalization + QG + result array',
                    native_campaign_qps_comparable=False)

    def done(self):
        self.index = None
