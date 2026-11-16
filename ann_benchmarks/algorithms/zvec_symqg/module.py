"""SymQG adapter using local_builder's persisted best-practice codec/router."""
import json
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import yaml
from py4bench import CoreSearcher

from ann_benchmarks.algorithms.base.module import BaseANN


class ZvecSymQG(BaseANN):
    def __init__(self, metric, dimension, views=0, query_mode="native-list"):
        if query_mode not in {"native-list", "legacy"}:
            raise ValueError("query_mode must be native-list or legacy")
        self.query_mode = query_mode
        self.metric, self.dimension = metric, dimension
        self.views = views or (2 if dimension in (100, 128) else 1)
        self.source = Path(os.environ.get(
            "ZVEC_SOURCE", str(Path.home() / "workspace/zvec/quantized_graph_exploration/zvec")))
        self.build = Path(os.environ.get("ZVEC_BUILD", str(self.source / "build-ann")))
        self.configs = self.source / "tools/core/bench/config"
        self.directory = None
        if (metric, dimension) not in {("angular", 100), ("euclidean", 128), ("euclidean", 960)}:
            raise ValueError("Best-practice presets only cover GloVe-100, SIFT-128 and GIST-960")
        self.dataset = {100: "glove", 128: "sift", 960: "gist"}[dimension]
        self.name = f"zvec-symqg({self.dataset},views={self.views})"

    def _build(self, config, name):
        filename = Path(self.directory.name) / (name + ".yaml")
        filename.write_text(yaml.safe_dump(config))
        subprocess.run([str(self.build / "bin/local_builder"), str(filename)], check=True)

    def _sift_base(self, X, vectors):
        # Both views must derive from the same raw graph, including when builds
        # run concurrently. Key the cache by input bytes, preset and executable.
        preset = self.configs / "sift_fp32_base_build.yaml"
        digest = hashlib.sha256(preset.read_bytes())
        digest.update((self.build / "bin/local_builder").read_bytes())
        digest.update(memoryview(np.ascontiguousarray(X)))
        cache = Path(os.environ.get("ZVEC_BASE_CACHE", "results/.zvec-base-cache")).resolve() / digest.hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / "raw.index"
        with (cache / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=cache) as stage:
                    base = yaml.safe_load(preset.read_text())
                    base["BuilderCommon"].update(BuildFile=str(vectors), IndexPath=str(Path(stage) / "work.index"),
                                                 DumpPath=str(Path(stage) / "raw.index"), MaxDocs=len(X))
                    self._build(base, "base")
                    os.replace(Path(stage) / "raw.index", target)
        return target

    def fit(self, X):
        self.done()
        self.directory = tempfile.TemporaryDirectory(prefix="ann-zvec-symqg-")
        root = Path(self.directory.name)
        # Use the official format converter; text retains FP32 round-trip precision.
        text = root / "train.txt"
        with text.open("w") as stream:
            for start in range(0, len(X), 4096):
                for i, row in enumerate(X[start:start + 4096], start):
                    stream.write(str(i) + ";" + " ".join(format(float(v), ".9g") for v in row) + "\n")
        vectors = root / "train.zvec.vecs"
        subprocess.run([str(self.build / "bin/txt2vecs"), "--input=" + str(text),
                        "--output=" + str(vectors), "--dimension=" + str(self.dimension),
                        "--method=L2"], check=True)
        if self.dataset == "sift":
            base_path = self._sift_base(X, vectors)
            config = yaml.safe_load((self.configs / f"sift_fp32_v{self.views}_build.yaml").read_text())
            config["BuilderCommon"]["InputIndex"] = str(base_path)
        else:
            config = yaml.safe_load((self.configs / f"{self.dataset}_fp32_build.yaml").read_text())
            config["BuilderCommon"].update(BuildFile=str(vectors), IndexPath=str(root / "work.index"), MaxDocs=len(X))
        self.path = str(root / "quantized.index")
        config["BuilderCommon"]["DumpPath"] = self.path
        config["QuantizedGraph"]["RotationViews"] = self.views
        self._build(config, "quantized")
        recall = yaml.safe_load((self.configs / f"{self.dataset}_fp32_recall_bench.yaml").read_text())
        self.params = json.loads(recall["IndexCommon"]["IndexConfig"])
        self.set_query_arguments(json.loads(recall["QueryConfig"]["QueryParam"]))

    def set_query_arguments(self, params):
        if isinstance(params, (int, float)):
            manifest = yaml.safe_load((self.configs / "annbenchmark_three_datasets.yaml").read_text())
            matches = [item for item in manifest["SelectedQueryArguments"]
                       if item["Dataset"] == self.dataset and item["Algorithm"] == "Zvec"
                       and item["Metric"] == "ann_recall" and item["Target"] == params
                       and item["Profile"] == f"{self.dataset}-zvec-v{self.views}"]
            if len(matches) != 1:
                raise ValueError(f"No unique best-practice preset for {self.dataset}, view {self.views}, target {params}")
            params = matches[0]["Parameters"]
        if not params:
            recall = yaml.safe_load((self.configs / f"{self.dataset}_fp32_recall_bench.yaml").read_text())
            params = json.loads(recall["QueryConfig"]["QueryParam"])
        self.__dict__.pop("query", None)
        self.searcher = None
        self.searcher = CoreSearcher(self.path, self.params, params)
        if self.query_mode == "native-list":
            # The three supported ANN datasets provide aligned contiguous FP32
            # queries. Keep native input validation and list materialization in
            # the timed call, avoiding redundant np.require and a NumPy output.
            self.query = self.searcher.search_list
        self.name = (f"zvec-symqg({self.dataset},views={self.views},"
                     f"query_mode={self.query_mode},query={params})")

    def query(self, q, n):
        return self.searcher.search(np.require(q, dtype=np.float32, requirements=["C", "A"]), n)

    def done(self):
        # A bound native method owns its searcher; drop it before cleanup/reload.
        self.__dict__.pop("query", None)
        self.searcher = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
