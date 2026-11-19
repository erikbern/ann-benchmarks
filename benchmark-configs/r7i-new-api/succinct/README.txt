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
DumpPath to local paths, build its executable with cmake --build build-r7i --target local_builder -j 12,
then run build-r7i/bin/local_builder base-DATASET.yaml,
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

Full revalidation is COMPLETE for this variant: 63 parameter slots, 945 measured rounds, 6,615,000 timed queries. All seven recall floors are reached on all three datasets. Large corpus/provider/index and
raw per-query files stay outside Git. Final provenance and summary references
are in the campaign result manifests; config check rejects legacy query fields.

Validated deliverables
  summary.csv / winners.json: actual ANN recall, strict ID recall, three repetition
    QPS values, their median, mean query microseconds, full winning query parameters.
  best-sift.json / best-gist.json / best-glove.json: executable winner replay inputs.
    Set a fresh ARTIFACT_ROOT, then use qg_bench.py check/prepare/build/measure
    with --config pointing to the chosen best-*.json (same --binary as above).
  validation/DATASET/: per-round measurements, complete source/index/provider
    manifests and references to every raw ID/score/latency/expansion file.
  validation/provenance.json and hardware.txt: actual source, binary, build flags,
    toolchain and hardware; raw large artifacts remain outside Git.
  validation/paired-correctness.json: historical IDs and distance ties separately.
  validation/capacity-validation.json: full-query paired capacity measurements.
  comparison.txt: both branches' measured recall tiers under this native protocol.
  validation/comparison.json: historical versus migrated versus tuned values;
    historical timings are explicitly NOT directly comparable.

Consistency checks (no external large files needed for metadata checks):
  python tools/core/bench/config/r7i/check_configs.py
Full audit on this machine, including every raw file hash and latency-derived QPS:
  python tools/core/bench/config/r7i/check_configs.py --raw

GloVe capacity caveat: ceil(1183514/1024)=1156 is the visited-bitmap threshold.
capacity=2048 uses the bitmap. capacity=128 uses a 1024-slot probing table,
which saturates for many high-recall queries; do not replace the high-recall
capacity with a small value just because the returned IDs stay unchanged.
The provider layout and Python/native timing boundary also changed from history;
new/old QPS differences are not a measurement of kernel optimization alone.

Measured source commit: 21c259867371586726e772145dcfa4b701af2d5f
ANN-Benchmarks test entry/config commit: 200f3c65be7e912a117d793bc1aff1359d2cab4f (c7ibench).
New and historical output comparison: SIFT 70000/70000 and GIST 7000/7000 ordered ID lists match; GloVe 69994/70000 match across the seven parameter groups. The six differing query/parameter cases, including actual distances, are listed separately. They are not classified as ties or claimed to be ID-identical. All within-new-version repeats are bitwise stable.
For all 126 campaign parameter slots, recall computed from native FP32 scores equals the independently recomputed FP64 distance-tolerance recall; see native-vs-independent-recall.json.

The tuning budget is at most 14 added parameter slots per dataset. SIFT has 18 distinct configurations in 21 slots (12 distinct additions); GIST/GloVe each have 21 distinct configurations (14 additions). Duplicate SIFT target slots are excluded from repeated winner selection.
