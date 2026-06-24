# Zeelin UIC-183

Zeelin UIC-183 is an angular/cosine approximate nearest-neighbor submission for ANN-Benchmarks.

The submitted implementation is:

`ZeelinLargecapPrefixbudgetM10BoundarytrimHnswgateIvf`

The submitted run group is:

`uic183_angular_singlequery_submission`

It uses a Faiss IVF-Flat inner-product index with an HNSW quantizer, normalized angular vectors, capped cell budgets, boundary-aware cell trimming, and compressed return prefixes. The submitted parameter set is intentionally fixed to the setting validated for the public angular benchmarks.

## Reproduce

```bash
python3 install.py --algorithm zeelin_uic183
python3 run.py --dataset glove-100-angular --algorithm zeelin-uic183 --runs 9
python3 run.py --dataset glove-25-angular --algorithm zeelin-uic183 --runs 9
python3 run.py --dataset nytimes-256-angular --algorithm zeelin-uic183 --runs 9
```

Batch-mode results can also be generated with `--batch`, but the primary submission target is the default single-query leaderboard.

## Local Official-Runner Validation

The following numbers were produced with ANN-Benchmarks `runner.run()` output HDF5 files and the official `knn` and QPS metric functions.

| Dataset | Recall | QPS | Public frontier QPS | Ratio |
|---|---:|---:|---:|---:|
| glove-100-angular | 0.10024 | 160733 | 51650 | 3.11x |
| glove-25-angular | 0.16277 | 212323 | 67354 | 3.15x |
| nytimes-256-angular | 0.11473 | 112771 | 55231 | 2.04x |
