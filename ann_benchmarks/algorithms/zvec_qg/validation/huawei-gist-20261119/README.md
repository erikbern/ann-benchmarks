# Huawei GIST investigation and correction — 2026-11-19

## Root cause of the misleading comparison

The previous report scanned `results/<dataset>/10/<algorithm>/` only. It omitted the dedicated vendor validation archive at `results/comprehensive-icelake-20261116/measurements/baseline/`. Its main-tree GIST Huawei records covered only 67 points on a different Q=1/max_edge=96 profile; the persisted tuned Q=3/max_edge=64 graph already had measured results around 7,157 / 4,024 / 989 QPS at 90% / 95% / 99% in the older two-repeat summary. The missing 99% entry was therefore an archive-selection error, not an algorithm limit.

The report loader now explicitly includes that dedicated archive and checks its dataset/algorithm metadata, query parameters, raw ID hashes and QPS against JSONL provenance. The corrected historical report is in `corrected-history/`. It retains the original campaign time cutoff; it does not silently mix this new Huawei campaign into historical records. This also corrects incomplete historical coverage for other dataset/algorithm combinations. The original incomplete comparison remains preserved and marked superseded.

## New measurements

Hardware and protocol match the new Zvec query protocol: Xeon Platinum 8369B, logical CPU 54, SMT sibling 55, NUMA memory bind 0, full GIST 1M FP32 corpus and all 1,000 queries, TopK=10, standard ANN Python-call timing. The graph is immutable and reused. Thirty-three predeclared query settings received a full warmup and one screening round. The original three high-recall settings plus two screened winners were confirmed in three fresh processes, each with full-query warmup then five rounds. Headline QPS is the median of three best-of-five values.

| Recall floor | Actual ANN recall | Strict ID recall | Huawei QPS | Three best5 observations | Query parameters [se,re,epsilon,approx,b,g] |
|---|---:|---:|---:|---|---|
| 90% | 90.37% | 88.19% | 6,999 | 6,985, 7,022, 6,999 | `[4, 8, 1.02, 0, 1, 3]` |
| 95% | 95.23% | 93.87% | 4,056 | 4,051, 4,056, 4,060 | `[6.4, 11.2, 1.03, 0, 1, 3]` |
| 99% | 99.04% | 98.56% | 1,010 | 1,010, 1,010, 1,012 | `[24.0, 120.0, 1.03, 0, 1, 3]` |

The 95% candidate improves 3.02% over the freshly remeasured original setting; the 99% candidate improves 4.70%. Their actual recalls differ (95.23% versus 95.48%, and 99.04% versus 99.00%), so these are improvements at a common floor, not at identical recall. The huge difference from the previous table comes primarily from omitted records, not these small query-parameter changes.

[Updated three-dataset reference table](current-comparison.txt) uses the newly confirmed Huawei values only for GIST; other competitor cells remain historical, with their original aggregation.

## Comparison with newly measured Zvec

The following two algorithms share the query timer and repetition protocol, on the same CPU but in separate isolated sessions. Different actual recalls are shown explicitly.

| Recall floor | Zvec QG QPS / actual recall | Huawei QSGNGT QPS / actual recall | QPS ratio |
|---|---:|---:|---:|
| 90% | 12,664 / 90.260% | 6,999 / 90.370% | 1.81x |
| 95% | 8,688 / 95.520% | 4,056 / 95.230% | 2.14x |
| 99% | 3,211 / 99.460% | 1,010 / 99.040% | 3.18x |

## Correctness and execution evidence

All 33 screening rounds and 75 confirmation rounds passed full-query ID/range/uniqueness, independently recomputed FP64 distances, ANN tolerance recall, strict ID recall, raw-file hashes and QPS recomputation. All three original configurations return exactly the historical IDs. All 15 confirmation rounds for each setting return identical IDs. Total timed queries: 108,000 (screen and confirmation); warmup queries are additional and excluded from QPS.

Initial preflight refused two resident threads before any timed query and restored all background jobs. The loader retains one helper thread. In the successful run every thread is pinned to CPU 54; per-round CPU accounting shows **zero total CPU seconds on non-main threads** across all 108 rounds, at the OS accounting resolution. This is one active query thread, not a multithreaded search. The failed preflight log and restoration record are retained. No result was selected based on a failed performance attempt.

Changing only query g=3 to g=0 on the tuned graph did not reproduce a multi-fold slowdown: the screened QPS ratios were close and recall changed slightly. This does not identify which build-time difference makes the main-tree Q=1 graph slower; a matched rebuild ablation would be needed to separate codec, adjacency and other construction choices. No instruction-set deficiency has been established by this test.

`plan.json` records actual persisted index headers as well as historical construction labels; these are not assumed interchangeable. It contains 1,309 data/index/adapter/runtime fingerprints. All were checked before each worker loaded the index. The borrowed full train array remains alive for the complete index/query lifetime. No graph was rebuilt or modified.

All 12 paused benchmark containers and two controllers were restored. `audit.json` records the checks. Large raw HDF5 files stay outside Git under `/home/xingrui/workspace/annbenchmark/ann-benchmarks/results/huawei-gist-revalidation-20261119-v2/`.

## Reproduction

Prepare the same fingerprinted graph, vendor Python wheel/libraries and dataset. Use a fresh output directory, copy this `plan.json` into it, and set `BENCH_ROOT`, `SOURCE` (succinct checkout), `ANN` (c7ibench checkout), `RUN` and `PYTHON`. The plan includes absolute artifact paths; when relocating, update paths while retaining hashes.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$BENCH_ROOT/vendor/runtime/usr/lib/x86_64-linux-gnu:$BENCH_ROOT/vendor/runtime/usr/lib/x86_64-linux-gnu/blas:$BENCH_ROOT/vendor/runtime/usr/lib/x86_64-linux-gnu/lapack:$BENCH_ROOT/vendor/huawei/lib:$BENCH_ROOT/vendor/huawei/lib/x86_64-linux-gnu"
"$PYTHON" "$SOURCE/tools/core/bench/qg_bench.py" isolate --manifest "$RUN/isolation.json" -- \
  numactl --physcpubind=54 --membind=0 "$PYTHON" "$ANN/scripts/diagnose_huawei_gist.py" \
    --bench-root "$BENCH_ROOT" --source "$SOURCE" --output "$RUN"
"$PYTHON" "$ANN/scripts/audit_huawei_gist.py" --run "$RUN"
```

The scripts reuse the existing vendor adapter and pinned index; the new code adds measurement/provenance and result selection, not Huawei kernel changes.
