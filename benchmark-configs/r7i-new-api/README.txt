Independent QG API benchmark — r7i local campaign, 2026-11-19

Actual hardware: Intel Xeon Platinum 8369B (Ice Lake), 64 logical CPUs,
32 physical cores, one NUMA node. The ann-benchmarks branch is named c7ibench;
that name is not a hardware identification. Query CPU 54, SMT sibling 55,
NUMA memory node 0. Historical configurations/results are preserved byte-for-byte
under history/20261118-845fe9ec; their new_api_revalidated flag is false.
g9i history is not evidence for this campaign.

Entry points
  tools/core/qg_bench.cc: native export/build/query executable (JSON via YAML parser).
  tools/core/bench/qg_bench.py: strict config loader, provider verification,
    index manifest binding, full-query measurement, validation and aggregation.
  tools/core/bench/qg_preflight.py: small correctness-only preflight.
  ann-benchmarks c7ibench/scripts/run_native_qg.py: runs the configurations
    committed on c7ibench through the selected Zvec source checkout.

Build from this checkout (CMake >=3.26, Ninja, GCC 11; Python numpy/h5py,
psutil/docker required; numactl for CPU and memory binding):
  git submodule update --init --recursive
  cmake -S . -B build-r7i -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON -DBUILD_TOOLS=ON \
    -DBUILD_C_BINDINGS=OFF -DBUILD_ZVEC_SHARED=OFF \
    -DBUILD_ZVEC_CORE_SHARED=OFF -DBUILD_ZVEC_AILEGO_SHARED=OFF -DENABLE_ICELAKE=ON
  cmake --build build-r7i --target qg_bench quantized_graph_index_test -j 12
  build-r7i/bin/quantized_graph_index_test
  python tools/core/bench/qg_preflight.py --binary "$PWD/build-r7i/bin/qg_bench" \
    --config-dir tools/core/bench/config/r7i --output /new/path/preflight

Run from ann-benchmarks c7ibench (set each path explicitly):
  export BENCH_ROOT=/path/to/ann-benchmarks-with-data-and-historical-indexes
  export ZVEC_SOURCE=/path/to/frozen-zvec-checkout
  export ARTIFACT_ROOT=/new/path/results/succinct
  python scripts/run_native_qg.py check --zvec-source "$ZVEC_SOURCE" \
    --bench-root "$BENCH_ROOT" --artifacts "$ARTIFACT_ROOT"
  python scripts/run_native_qg.py prepare --zvec-source "$ZVEC_SOURCE" \
    --bench-root "$BENCH_ROOT" --artifacts "$ARTIFACT_ROOT"
  python scripts/run_native_qg.py build --zvec-source "$ZVEC_SOURCE" \
    --bench-root "$BENCH_ROOT" --artifacts "$ARTIFACT_ROOT"
  python "$ZVEC_SOURCE/tools/core/bench/qg_bench.py" isolate \
    --manifest "$ARTIFACT_ROOT/isolation.json" -- \
    python scripts/run_native_qg.py measure --zvec-source "$ZVEC_SOURCE" \
      --bench-root "$BENCH_ROOT" --artifacts "$ARTIFACT_ROOT"
Repeat with exploration's frozen source and a DIFFERENT ARTIFACT_ROOT.
Do not run the two variants' formal measurements concurrently.
The isolation command stops benchmark controllers, pauses all ann-benchmarks
containers, records restoration identities, checks for compilation, then restores
on completion/failure. Interrupted background timings must be excluded from future
plots. Recovery after a hard kill:
  python "$ZVEC_SOURCE/tools/core/bench/qg_bench.py" restore \
    --manifest "$ARTIFACT_ROOT/isolation.json"

The wrapper resolves ${BENCH_ROOT}/${ARTIFACT_ROOT} into resolved.json, actually
read by qg_bench. A standalone snapshot query diagnostic is:
  build-r7i/bin/qg_bench query /path/to/resolved.json n64 /new/path/query-output
Use the Python measure entry for formal timings; it verifies the manifest contract.
The n32 profile exists only for GloVe. Different neighbor counts/codecs/views have
separate freshly built snapshots. Router and codec rotation_views must agree.

Data/provider contract
Full HDF5 train/test sets (SIFT 1M x128/10000; GIST 1M x960/1000;
GloVe 1183514x100/10000), FP32, top10. Existing immutable HNSW base graphs
are hash-checked and exported with the new source's entity reader. Every exported
vector is checked against the original HDF5 row (normalized once for cosine),
all IDs are a permutation, and adjacency is range checked. No old QG snapshot
is loaded: train_codec, train_router, prebuild, dump and load all use the new API.
CSR files are little-endian: vectors FP32[count,dim], IDs uint64[count],
offsets uint64[count+1], edges uint32[offsets[-1]]. Owned arrays outlive queries.
Cosine corpus vectors were normalized by the historical base builder and are
validated, not normalized again; raw queries are normalized once during prepare.
Exact scoring uses SIMD SquaredEuclidean or 1 + negative InnerProduct; independent
FP64 L2/cosine validation uses raw HDF5 data, outside the query timer.
Rebuilding a base graph is OPTIONAL and changes the experiment: base-*.yaml are
plain HNSW local_builder recipes (no QG keys). Set their BuildFile/IndexPath/
DumpPath to local paths, run build-r7i/bin/local_builder base-DATASET.yaml,
then update source_index and source_sha256 in a NEW campaign configuration.
The archived native base graphs are required for exact reproduction of this run;
parallel HNSW rebuilding is not claimed to produce identical graph topology.

Protocol and comparisons
Each parameter slot: full-query warmup, five measured rounds; three fresh process
repetitions loading the same immutable snapshot. Each round times each query via
steady_clock; QPS=1/mean latency. Repetition QPS=max(five rounds); headline QPS=
median(three repetition QPS). Mean microseconds=1e6/headline QPS (the selected
median repetition's best round). No preflight numbers enter formal summaries.
All result IDs, scores, latencies, expansion and exact-comparison counts persist
per round, with fingerprints. Every round's ANN and strict ID recalls are saved.
Repeated queries must return identical IDs AND scores across all measured rounds.
ANN recall uses raw-data Euclidean distance (sqrt of squared L2), or cosine,
compared to HDF5's tenth true distance +0.001. Strict ID recall is separate.
Each target retains frozen historical search parameters first. Capacity is seeded
from historical ef_search because the old HnswQueryAdapter reset used
max(topk,ef), while new QuantizedGraphQuery uses max(1,topk,capacity). This is
also tested by exhaustive small graphs and paired full-query capacity changes;
parameter equality alone is not asserted to prove result identity.
Tuning is predeclared: 14 additional parameter slots per dataset, two per recall
target. Winners are measured points, never interpolated. Repeated target slots
can share parameters (SIFT 85/90); compare first occurrence for identical configs.
The native timer includes context reset, QG search and its exact callbacks,
result sorting, external ID mapping and buffer copy. It excludes Python calls,
input conversion/normalization, file I/O and independent distance verification.
Historical Python-call timing and background load differ; the old/new table is
NOT a direct kernel speedup comparison. Reusing one snapshot for repetitions
does not measure random variation from rebuilding indexes.

Revalidation is COMPLETE. Results, executable best-*.json presets, and metadata/raw checks are in succinct/ and exploration/. Large corpus/provider/index and
raw per-query files stay outside Git. Final provenance and summary references
are in the campaign result manifests; config check rejects legacy query fields.

One-command full reproduction from the c7ibench checkout (fresh output directory):
  BENCH_ROOT=/home/xingrui/workspace/annbenchmark/ann-benchmarks \
  SUCCINCT_SOURCE=/home/xingrui/workspace/zvec/r7i-new-api-succinct \
  EXPLORATION_SOURCE=/home/xingrui/workspace/zvec/r7i-new-api-exploration \
  ARTIFACT_ROOT=/home/xingrui/workspace/annbenchmark/ann-benchmarks/results/new-replay \
  bash benchmark-configs/r7i-new-api/reproduce.sh

The root sift/gist/glove JSON files are the actual frozen campaign inputs.
Variant subdirectories contain the full validation archive and selected runnable
best-sift.json, best-gist.json, best-glove.json configurations. The test entry
commit was 200f3c65be7e912a117d793bc1aff1359d2cab4f. Raw large artifacts remain
under the recorded external result directory; no dataset/index is committed.

Review checks, without the large external artifacts:
  python benchmark-configs/r7i-new-api/succinct/check_configs.py
  python benchmark-configs/r7i-new-api/exploration/check_configs.py
Append --raw on the measurement machine to recheck all 9450 raw file hashes
and recompute every round QPS. Summary tables use median of three best-of-five
repetitions. Historical Python-call timings are not directly comparable.
