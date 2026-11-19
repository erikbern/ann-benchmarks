# Formal zvec_qg Python-entry measurements — 2026-11-19

**Historical comparator correction:** the original table scanned only the main
results tree and missed the dedicated vendor revalidation archive. In particular,
Huawei GIST's 1,315/274/no-99% values were not its best known results. Zvec's own
formal measurements remain valid. Use the [corrected comparison and new Huawei
GIST measurements](../huawei-gist-20261119/README.md); the original table is kept
as an audit record and must not be used as a complete vendor comparison.

Succinct source `898e87b08591940dd2896191342e6481730fc6da`, validated core `21c259867371586726e772145dcfa4b701af2d5f`; ANN adapter `fca371990a1544602b18429b39fa4f03f167ab16` plus the fingerprinted `scripts/measure_zvec_qg.py` added in this delivery. No core or adapter change during measurement.

Actual CPU: Xeon Platinum 8369B (Ice Lake). Single query/thread, CPU 54, SMT sibling 55, NUMA memory bind 0. Twelve benchmark containers and two controllers were paused and restored successfully. Nine fresh measurement processes have recorded affinity, thread count and NUMA maps.

Each of 21 settings uses all training vectors and the entire query set: SIFT 10,000, GIST 1,000, GloVe 10,000. One full-query runner warmup per setting and process; three process repetitions × five rounds. All 315 rounds and 2,205,000 timed queries passed ID, distance, recall and raw-fingerprint checks. All three repetitions reuse the same snapshots, so this does not cover reconstruction randomness.

The timer is the unmodified ANN-Benchmarks `run_individual_query`: `time.time()` immediately around `algo.query`. Python input handling, cosine normalization, native QG, and result-array creation are included. Independent distance computation is outside timing. It still occurs between timed queries, just as in the standard runner. The primary QPS is median(three best-of-five QPS values); mean microseconds = 1e6 / primary QPS.

## Measured results

| Dataset | Recall floor | Actual ANN recall | Repeat 0 best5 | Repeat 1 best5 | Repeat 2 best5 | Main QPS | Mean µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 30% | 31.954% | 67,544 | 67,942 | 67,078 | 67,544 | 14.81 |
| SIFT | 50% | 51.806% | 58,865 | 59,117 | 58,368 | 58,865 | 16.99 |
| SIFT | 70% | 73.793% | 49,299 | 49,363 | 48,896 | 49,299 | 20.28 |
| SIFT | 85% | 85.474% | 42,571 | 42,740 | 42,378 | 42,571 | 23.49 |
| SIFT | 90% | 90.260% | 39,277 | 39,478 | 39,159 | 39,277 | 25.46 |
| SIFT | 95% | 95.173% | 33,005 | 33,228 | 32,925 | 33,005 | 30.30 |
| SIFT | 99% | 99.058% | 22,452 | 22,547 | 22,385 | 22,452 | 44.54 |
| GIST | 30% | 31.680% | 34,770 | 35,097 | 35,257 | 35,097 | 28.49 |
| GIST | 50% | 51.910% | 27,489 | 28,076 | 28,012 | 28,012 | 35.70 |
| GIST | 70% | 70.090% | 22,038 | 22,384 | 22,393 | 22,384 | 44.67 |
| GIST | 85% | 86.630% | 14,977 | 14,845 | 15,331 | 14,977 | 66.77 |
| GIST | 90% | 90.260% | 12,664 | 12,583 | 12,964 | 12,664 | 78.97 |
| GIST | 95% | 95.520% | 8,688 | 8,616 | 8,933 | 8,688 | 115.10 |
| GIST | 99% | 99.460% | 3,211 | 3,181 | 3,320 | 3,211 | 311.45 |
| GLOVE | 30% | 32.275% | 41,914 | 41,808 | 42,035 | 41,914 | 23.86 |
| GLOVE | 50% | 50.089% | 35,923 | 35,864 | 36,138 | 35,923 | 27.84 |
| GLOVE | 70% | 70.266% | 26,750 | 26,623 | 26,770 | 26,750 | 37.38 |
| GLOVE | 85% | 85.265% | 16,492 | 16,353 | 16,438 | 16,438 | 60.83 |
| GLOVE | 90% | 90.010% | 12,690 | 12,616 | 12,685 | 12,685 | 78.83 |
| GLOVE | 95% | 95.026% | 8,153 | 8,058 | 8,108 | 8,108 | 123.34 |
| GLOVE | 99% | 99.001% | 3,491 | 3,448 | 3,469 | 3,469 | 288.24 |

## Historical comparison boundary

[table.txt](table.txt) and [comparison.json](comparison.json) compare these results with existing Huawei QSGNGT and 01AI Descartes records from the main `results/<dataset>/10/<algorithm>/` tree. Only completed single-query, five-run, full-query records with modification time before this campaign are included. Invalid shapes, IDs, times and distances are rejected; selected historical winners are independently distance-checked against the raw corpus. No competitor performance was rerun.

Huawei GIST has only 67 valid points in this selected archive and no point reaching 99%; this is missing coverage, not a claim that the algorithm cannot attain it. Results outside this archive are not included. Different actual recalls at a common floor are reported in comparison.json.

Competitor values are historical best-of-five measurements, often under background workloads; Zvec values are new isolated median-of-three measurements. They are a reference comparison, not a same-batch ranking or a controlled kernel-speedup claim. Zvec repetition 0 best-of-five is included as a separately labeled aggregation match; we do not take the highest QPS across repetitions.

## Evidence and reproduction

See [run.json](run.json) for executable/preset/source fingerprints and the exact measurement command; [rounds.json](rounds.json) for each round, [audit.json](audit.json) for raw-file verification, and [restore-verified.json](restore-verified.json) for restoration. Compiler and linked-library fingerprints are in the parent validation build manifests and run.json artifact. [summary.csv](summary.csv) includes both ANN tolerance and strict ID recall.

Use the formal Python-entry commands in [the adapter README](../../README.md). The large raw HDF5 records remain outside Git at:

`/home/xingrui/workspace/annbenchmark/ann-benchmarks/results/r7i-ann-formal-20261119/measurements/`

The previous native-timer campaign is preserved unchanged and is not relabeled as Python-entry measurements.
