# r7i adapter validation — 2026-11-19

The subsequent isolated Python-entry performance campaign is now complete:
[formal measurements and comparison](validation/formal-20261119/README.md).
The correctness-only runs described below remain separate from those timings.

This validates the unified `zvec_qg` entry and colocated configurations on actual
Intel Xeon Platinum 8369B hardware. It follows the organization of g9ibench
`07fcd9bed0b0`, using this machine's independently built QG snapshots and CSR
provider. No g9i results or legacy QG snapshots are reused.

## Correctness coverage

Both succinct and exploration: all three complete training corpora, seven recall
settings per dataset, and two passes over every query (10,000 SIFT, 1,000 GIST,
10,000 GloVe). This is 42 settings and 588,000 checked queries across two variants.

- Exactly 10 legal, unique external IDs per query.
- Independent FP64 distance verification against the original train/query data.
- ANN distance-tolerance recall (tenth ground-truth distance +0.001) and strict
  ID recall reported separately in JSON.
- Every query expands QG nodes. The bridge contains no HNSW search invocation.
- Repeated queries return bitwise-identical IDs and scores.
- All returned IDs and scores are bitwise identical to the corresponding native
  campaign measurements on these same new snapshots.
- Invalid dimensions, nonfinite queries, wrong TopK and batch queries are rejected.
- Config boundary checks reject retired keys, incompatible rotation views,
  negative capacity, nonfinite options, metric mismatch and missing environment.

Actual distance-tolerance recalls are identical for both variants:

| Dataset | ≥30% | ≥50% | ≥70% | ≥85% | ≥90% | ≥95% | ≥99% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT | 31.954% | 51.806% | 73.793% | 85.474% | 90.260% | 95.173% | 99.058% |
| GIST | 31.680% | 51.910% | 70.090% | 86.630% | 90.260% | 95.520% | 99.460% |
| GloVe | 32.275% | 50.089% | 70.266% | 85.265% | 90.010% | 95.026% | 99.001% |

## Standard ANN CLI integration

`run.py --local --algorithm zvec_qg --parallelism 1 --runs 1` completed all three
datasets with the generated definitions and resolved succinct profiles: 21 HDF5
results and 147,000 queries. Every stored neighbor ID matches the native reference;
ANN's independently recomputed distances give the recalls above. This covers
algorithm discovery, environment/profile loading, snapshot switching, result
serialization and all seven query settings.

These are correctness runs on CPU 54 with background workloads active. No jobs
were suspended. THP policy was the native campaign's `prepare`, with unknown
coverage warnings from the loader; strict g9i `require` admission was not adopted.
No QPS from these runs enters a performance table. They do not constitute a new
isolated 3 × 5 Python-timing campaign. Original native campaign results and their
timing/provenance remain in `benchmark-configs/r7i-new-api/` unchanged.

## Evidence and reproduction

- [Succinct full-query record](validation/succinct.json).
- [Exploration full-query record](validation/exploration.json).
- [ANN CLI result fingerprints and recalls](validation/cli.json).
- [Config rejection checks](validation/config-boundaries.json).
- `validation/build-{succinct,exploration}.json`: exact bridge compiler and link commands.

Each full-query record includes native source/validated commit, bridge/library
fingerprints, Python validator/adapter fingerprints, preset fingerprints, and
raw result paths/hashes. Raw NPZ/HDF5 results, provider files, QG snapshots and
binaries remain outside Git under:

`/home/xingrui/workspace/annbenchmark/ann-benchmarks/results/r7i-ann-adapter-20261119/`

Use [README commands](README.md) to build, resolve profiles and repeat full-query
validation. Use `--runs 1` for the recorded CLI integration check. The full-query
script writes `passed=true` only after all datasets and boundary checks finish.
