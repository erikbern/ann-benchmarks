# c7ibench: independent Zvec QG

The algorithm layout follows `g9ibench` commit `07fcd9bed0b0`:

- One ANN-Benchmarks algorithm name: `zvec_qg`.
- [Adapter, runtime build and commands](ann_benchmarks/algorithms/zvec_qg/README.md).
- Runnable presets colocated in `ann_benchmarks/algorithms/zvec_qg/configs/r7i/`
  (succinct, the default), with `configs/r7i-exploration/` for controlled comparisons.
- [Native campaign and immutable measurement archive](benchmark-configs/r7i-new-api/README.txt).

Actual measured hardware is Xeon Platinum 8369B, Ice Lake, 64 logical CPUs,
32 cores and one NUMA node. Branch/directory names do not identify an instance SKU.
This machine uses the already validated CSR providers and newly built independent
QG snapshots. It does not use the g9i server's legacy container, graph parameters,
THP admission policy, build paths or performance results.

The existing native campaign remains a separate, complete 3-repeat × 5-round
experiment. Adding the Python adapter does not reclassify those QPS numbers as
ANN's Python-call timings. The new adapter regression checks correctness on every
query and records no formal performance ranking. See the adapter's validation
record for coverage and the exact runtime provenance.

```bash
python scripts/check_zvec_qg_configs.py
python benchmark-configs/r7i-new-api/succinct/check_configs.py
python benchmark-configs/r7i-new-api/exploration/check_configs.py
```
