# Zvec QG on c7ibench

[Full-query and CLI validation](VALIDATION.md)

Select `--local --algorithm zvec_qg`. This uses the independent
`QuantizedGraphIndex` / `QuantizedGraphQuery` API. The owned FP32 CSR provider,
ID mapping, exact distance callback and new QG snapshots are those validated in
the r7i native campaign. No HNSW search or retired QG dispatcher is called.

Like g9ibench, runnable presets are colocated with the adapter:

- `configs/r7i/{sift,gist,glove}.json`: succinct, default.
- `configs/r7i-exploration/{sift,gist,glove}.json`: exploration.

Settings 0–6 correspond to recall floors 30%, 50%, 70%, 85%, 90%, 95%, 99%.
These are measured native-campaign winners; a floor is not an exact recall.
GloVe switches n32/n64 QG snapshots outside the query timer. Encoding and router
rotation views agree, and every corpus/provider/snapshot is fingerprint checked.
All search keys use the new names, including explicit capacity and scan_limit.
The consistency checker ties presets to archived winning parameters and artifacts.

## Build the adapter

Use Python with numpy, h5py, pybind11, PyYAML and the ANN-Benchmarks dependencies.
Supply the previously built and validated Zvec source tree. The build script
checks the measured source revision and native executable fingerprint, derives
compiler/link options from its CMake/Ninja build, freshly compiles the bridge,
and records all linked library fingerprints. Core changes require new validation.
The existing native build uses GCC 11.4, Release `-O3 -DNDEBUG -march=icelake-server`.

```bash
ANN=/path/to/c7ibench
BENCH_ROOT=/path/to/ann-benchmarks-with-data
PYTHON="$BENCH_ROOT/.venv/bin/python"
SOURCE=/path/to/r7i-new-api-succinct
RUN=/new/path/ann-qg-run
export PATH="$BENCH_ROOT/.venv/bin:$PATH"
cd "$ANN"
"$PYTHON" scripts/build_zvec_qg.py --source "$SOURCE" \
  --variant succinct --output "$RUN/runtime"
export ZVEC_QG_ARTIFACT="$RUN/runtime/artifact.json"
# Directory containing sift/, gist/ and glove/ provider files and new snapshots.
export ZVEC_QG_INDEX_ROOT="$BENCH_ROOT/results/r7i-new-api-20261119/succinct"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QG_THP_POLICY=prepare
unset QG_THP_MIN_COVERAGE
"$PYTHON" scripts/prepare_zvec_qg_profiles.py --platform r7i --output "$RUN/profiles"
```

To reproduce missing native providers/snapshots, use the complete build, export,
QG training and measurement commands in
[`benchmark-configs/r7i-new-api/reproduce.sh`](../../../benchmark-configs/r7i-new-api/reproduce.sh).
Large data, indexes, shared libraries and raw results are not committed to Git.
The bridge is load-only: `fit` verifies the full training corpus and files, owns
the provider arrays and loads the first QG snapshot. ANN's build time is therefore
validation/load time, not graph construction. The source's native build entry
remains available for actual `QuantizedGraphBuildOptions` training and snapshots.

## Validate and run

```bash
"$PYTHON" scripts/validate_zvec_qg.py --artifact "$ZVEC_QG_ARTIFACT" \
  --data "$BENCH_ROOT/data" --variant succinct --output "$RUN/correctness"
mkdir -p "$RUN/cli"
ln -s "$BENCH_ROOT/data" "$RUN/cli/data"
ln -s "$ANN/logging.conf" "$RUN/cli/logging.conf"
cd "$RUN/cli"
numactl --physcpubind=54 --membind=0 "$PYTHON" "$ANN/run.py" \
  --local --algorithm zvec_qg --definitions "$RUN/profiles/definitions" \
  --dataset sift-128-euclidean --parallelism 1 --runs 5
```

For GIST/GloVe select `gist-960-euclidean` / `glove-100-angular`. Run datasets
serially. CPU 54 has SMT sibling 55, both on NUMA node 0. Avoid other benchmarks,
builds and compilation during formal timings; the native campaign's `isolate`
wrapper can pause and restore the existing controllers/containers around a command.
Keep its restoration manifest and restore after failures. Use a fresh run directory
because ANN's standard CLI skips existing results by default.

For exploration build a separate runtime with `--variant exploration`, set
`ZVEC_QG_INDEX_ROOT` to its separate artifact directory, and prepare profiles with
`--platform r7i-exploration`. Pass `--variant exploration` to the correctness script.
The generated definitions select that variant under the same `zvec_qg` name.
Keep variant result directories separate; the adapter refuses mismatched runtimes.

## Measurement boundary

Python query timings include input validation/conversion, FP64 cosine query
normalization once, the native search and exact callbacks, sorting/ID mapping,
and Python result-array materialization. Original cosine provider vectors are
already normalized and are not normalized again. Independent distance checking
by ANN's runner is outside query timing. Batch/concurrent query calls are rejected.

ANN's CLI reports best-of-runs and does not implement the native campaign's full
warmup plus three repetitions of five rounds. Its output must be labeled separately
from the native median-of-three-best-of-five QPS. A CLI integration test or the
correctness script is not a new formal performance campaign.

The correctness script uses all train rows and all queries, two query passes per
setting, independent FP64 distance checks, distance-tolerance and strict ID recalls,
QG expansion evidence, and bitwise comparison against archived native IDs/scores.
Per-query artifacts are written outside Git; the small validation manifest is
archived alongside this adapter. It makes no claim about new Python-entry QPS.

ANN's incremental memory field includes the first snapshot only; GloVe can load a
different snapshot per setting. Additional result fields identify the active
profile and total process RSS. The adapter owns its provider; ANN retains the
training corpus for distance verification, so total RSS includes both allocations.

The historical native measurement archive under `benchmark-configs/r7i-new-api/`
is unchanged. No g9i timing or historical Python timing is relabeled as this run.

## Formal Python-entry measurements

[Completed 2026-11-19 results, per-round evidence and historical comparison](validation/formal-20261119/README.md).

[Corrected vendor archive coverage and new Huawei GIST confirmation](validation/huawei-gist-20261119/README.md).

`scripts/measure_zvec_qg.py` calls the unmodified ANN `run_individual_query`
function once per round, preserving every round instead of retaining only the
last round's neighbors. Each setting gets one complete runner warmup and five
measured rounds. Three fresh Python processes per dataset reload the same index;
the primary QPS is the median of the three best-of-five QPS values. It checks
the full dataset fingerprint, IDs, independent distances and both recall metrics
each round. The first repetition's best-of-five is also reported separately for
comparison with historical ANN summaries; it is not the highest of 15 rounds.

With the environment above and `SOURCE` pointing to succinct:

```bash
"$PYTHON" "$SOURCE/tools/core/bench/qg_bench.py" isolate \
  --manifest "$RUN/isolation.json" -- \
  numactl --physcpubind=54 --membind=0 "$PYTHON" "$ANN/scripts/measure_zvec_qg.py" \
    --artifact "$ZVEC_QG_ARTIFACT" --data "$BENCH_ROOT/data" \
    --source "$SOURCE" --output "$RUN/measurements"
"$PYTHON" "$ANN/scripts/report_zvec_qg.py" --run "$RUN" --bench-root "$BENCH_ROOT"
```

The isolation wrapper restores stopped controllers and paused containers on
completion or failure. After an external hard kill, use its `restore --manifest`
command. Interrupted background timings must not enter future comparisons.
The report audits all 315 raw HDF5 files and snapshots only valid, already-finished
Huawei/01AI results predating this campaign. It includes both the standard results
tree and the dedicated `comprehensive-icelake-20261116/measurements/baseline`
archive, validating the latter's JSONL provenance and ID hashes. It does not rerun these algorithms.
Their historical best-of-five values and background conditions remain distinct
from the new isolated median-of-three results.
