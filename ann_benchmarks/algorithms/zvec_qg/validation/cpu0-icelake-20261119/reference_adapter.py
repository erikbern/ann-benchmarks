"""Independent QG queries on an immutable, fingerprinted legacy index container."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from ..base.module import BaseANN


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_config(path):
    return json.loads(Path(os.path.expandvars(str(path))).read_text())


def load_profile(profile, metric, dimension):
    if profile == "g9i":
        presets = Path(__file__).parent / "configs" / "g9i"
        matches = [read_config(path) for path in sorted(presets.glob("*.json"))]
        matches = [item for item in matches
                   if item["metric"] == metric and item["dimension"] == dimension]
        if len(matches) != 1:
            raise ValueError("No unique g9i preset for this metric/dimension")
        result = matches[0]
    else:
        result = read_config(profile)
    result["index_path"] = os.path.expandvars(result["index_path"])
    if "$" in result["index_path"]:
        raise ValueError("Set the index environment variable: " + result["index_path"])
    return result


class ZvecQuantizedGraph(BaseANN):
    """Load-only adapter; fit verifies the supplied corpus rather than rebuilding."""

    def __init__(self, metric, dimension, profile, artifact):
        self.profile = load_profile(profile, metric, dimension)
        self.artifact = read_config(artifact)
        self.index = None
        if metric != self.profile["metric"] or dimension != self.profile["dimension"]:
            raise ValueError("Dataset metric/dimension does not match the frozen profile")
        self.dimension = dimension
        self.name = "zvec_qg " + self.profile["dataset"]
        for path, expected in self.artifact["runtime_sha256"].items():
            if digest(path) != expected:
                raise ValueError("Runtime fingerprint mismatch: " + path)
        spec = importlib.util.spec_from_file_location("_direct_core", self.artifact["extension"])
        self.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.core)

    def fit(self, X):
        if self.index is not None:
            raise RuntimeError("Close the existing index before loading another")
        if X.dtype != np.float32 or not X.flags.c_contiguous:
            raise ValueError("The frozen corpus must be contiguous float32")
        if list(X.shape) != self.profile["train_shape"]:
            raise ValueError("Training corpus shape mismatch")
        if hashlib.sha256(memoryview(X)).hexdigest() != self.profile["train_sha256"]:
            raise ValueError("Training corpus fingerprint mismatch")
        path = self.profile["index_path"]
        if digest(path) != self.profile["index_sha256"]:
            raise ValueError("Index fingerprint mismatch: " + path)
        self.index = self.core.Index(path, json.dumps(self.profile["storage"]), self.dimension)

    def set_query_arguments(self, options):
        if isinstance(options, int):
            if options < 0 or options >= len(self.profile["settings"]):
                raise ValueError("Query setting index is out of range")
            options = self.profile["settings"][options]["options"]
        required = {"capacity", "margin", "bucket_width", "max_expansions", "route_beam", "uncertainty_bonus"}
        if set(options) != required or not isinstance(options["capacity"], int) or options["capacity"] <= 0:
            raise ValueError("Expected a positive capacity and the five native search options")
        if self.index is None:
            raise RuntimeError("Load the index before configuring queries")
        # These names belong only to the private bridge parser. The bridge calls
        # QuantizedGraphQuery directly; it never passes them to HNSW IndexFactory.
        bridge = {"ef_search": options["capacity"]}
        bridge.update({"quantized_graph_" + key: value for key, value in options.items() if key != "capacity"})
        self.index.configure(json.dumps(bridge))

    def query(self, q, n):
        return self.index.search(q, n)

    def get_additional(self):
        return {"source_commit": self.artifact["source_commit"],
                "entry": "QuantizedGraphIndex", "query_threads": 1,
                "index_disk_bytes": Path(self.profile["index_path"]).stat().st_size,
                "borrowed_corpus_bytes": 0}

    def done(self):
        if self.index is not None:
            self.index.close()
            self.index = None
