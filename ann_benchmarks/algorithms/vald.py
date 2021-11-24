from __future__ import absolute_import

import atexit
import subprocess
import urllib.error
import urllib.request

import grpc
import yaml
from ann_benchmarks.algorithms.base import BaseANN

from vald.v1.vald import insert_pb2_grpc, search_pb2_grpc
from vald.v1.agent.core import agent_pb2_grpc
from vald.v1.payload import payload_pb2


default_server_config = {
    'version': 'v0.0.0',
    'logging': {
        'logger': 'glg',
        'level': 'info',
        'format': 'raw'
    },
    'server_config': {
        'servers': [
            {
                'name': 'agent-grpc',
                'host': '127.0.0.1',
                'port': 8082,
                'mode': 'GRPC',
                'probe_wait_time': '3s',
                #'grpc': {
                #    'bidirectional_stream_concurrency': 1
                #},
                "network": "unix",
                "socket_path": "/var/run/vald.sock"
            }
        ],
        'health_check_servers': [
            {
                'name': 'readiness',
                'host': '127.0.0.1',
                'port': 3001,
                'mode': '',
                'probe_wait_time': '3s',
                'http': {
                    'shutdown_duration': '5s',
                    'handler_timeout': '',
                    'idle_timeout': '',
                    'read_header_timeout': '',
                    'read_timeout': '',
                    'write_timeout': ''
                }
            }
        ],
        'startup_strategy': ['agent-grpc', 'readiness'],
        'shutdown_strategy': ['readiness', 'agent-grpc'],
        'full_shutdown_duration': '600s',
        'tls': {
            'enabled': False,
        }
    },
    'ngt': {
        'enable_in_memory_mode': True,
        'default_pool_size': 10000,
        'default_epsilon': 0.01,
        'default_radius': -1.0,
        #'vqueue': {
        #    'insert_buffer_size': 100,
        #    'insert_buffer_pool_size': 1000,
        #    'delete_buffer_size': 100,
        #    'delete_buffer_pool_size': 1000
        #}
    }
}

grpc_opts = [
    ('grpc.keepalive_time_ms', 1000 * 10),
    ('grpc.keepalive_timeout_ms', 1000 * 10),
    ('grpc.max_connection_idle_ms', 1000 * 50)
]

metrics = {'euclidean': 'l2', 'angular': 'cosine'}


class Vald(BaseANN):
    def __init__(self, metric, object_type, params):
        self._param = default_server_config
        self._ngt_config = {
            'distance_type': metrics[metric],
            'object_type': object_type,
            'search_edge_size': int(params['searchedge']),
            'creation_edge_size': int(params['edge']),
            'bulk_insert_chunk_size': int(params['bulk'])
        }
        #self._address = 'localhost:8082'
        self._address = 'unix:///var/run/vald.sock'

    def fit(self, X):
        dim = len(X[0])
        self._ngt_config['dimension'] = dim
        self._param['ngt'].update(self._ngt_config)
        with open('config.yaml', 'w') as f:
            yaml.dump(self._param, f)

        cfg = payload_pb2.Insert.Config(skip_strict_exist_check=True)
        vectors = [
            payload_pb2.Insert.Request(
                vector=payload_pb2.Object.Vector(id=str(i), vector=x.tolist()),
                config=cfg) for i, x in enumerate(X[:100])]

        p = subprocess.Popen(['/go/bin/ngt', '-f', 'config.yaml'])
        atexit.register(lambda: p.kill())

        while True:
            try:
                with urllib.request.urlopen('http://localhost:3001/readiness') as response:
                    if response.getcode() == 200:
                        break
            except (urllib.error.HTTPError, urllib.error.URLError):
                pass

        channel = grpc.insecure_channel(self._address, grpc_opts)
        istub = insert_pb2_grpc.InsertStub(channel)
        for _ in istub.StreamInsert(iter(vectors)):
            pass

        astub = agent_pb2_grpc.AgentStub(channel)
        astub.CreateIndex(
            payload_pb2.Control.CreateIndexRequest(
                pool_size=10000))

    def set_query_arguments(self, epsilon):
        self._epsilon = epsilon - 1.0
        channel = grpc.insecure_channel(self._address, grpc_opts)
        self._stub = search_pb2_grpc.SearchStub(channel)

    def query(self, v, n):
        cfg = payload_pb2.Search.Config(num=n, radius=-1.0, epsilon=self._epsilon, timeout=3000000)
        response = self._stub.Search(payload_pb2.Search.Request(vector=v.tolist(), config=cfg))
        return [int(result.id) for result in response.results]

    def __str__(self):
        return 'Vald(%d, %d, %d, %1.3f)' % (
            self._ngt_config['creation_edge_size'],
            self._ngt_config['search_edge_size'],
            self._ngt_config['bulk_insert_chunk_size'],
            self._epsilon + 1.0
        )
