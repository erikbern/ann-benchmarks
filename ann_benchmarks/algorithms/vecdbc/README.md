# ann-benchmarks adapter for vecdb

These three files make vecdb runnable inside
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), the standard
ANN evaluation harness. They are kept here for versioning; to actually run or
submit, copy them into a clone of ann-benchmarks.

## Files
- `module.py` — `Vecdb(BaseANN)` wrapper (fit / set_query_arguments / query).
  Handles euclidean directly and angular by L2-normalizing (cosine == L2 on
  unit vectors; vecdb uses squared L2).
- `config.yml` — parameter grid (M, efConstruction, and the ef sweep).
- `Dockerfile` — installs `vecdbc` from PyPI (compiles the C with a portable
  AVX2 baseline) on top of the base ann-benchmarks image.

## To run locally
```sh
git clone https://github.com/erikbern/ann-benchmarks
cd ann-benchmarks
mkdir -p ann_benchmarks/algorithms/vecdb
cp /path/to/this/dir/{module.py,config.yml,Dockerfile} ann_benchmarks/algorithms/vecdb/
pip install -r requirements.txt
python install.py --algorithm vecdb          # builds the Docker image
python run.py --dataset sift-128-euclidean --algorithm vecdb
python plot.py --dataset sift-128-euclidean
```

## To submit upstream
Open a PR to erikbern/ann-benchmarks adding `ann_benchmarks/algorithms/vecdb/`
with these files, and add `vecdb` to the CI test list (`.github/workflows`)
for the `random-xs-*` datasets so the maintainers' CI exercises it.

Note: ann-benchmarks runs single-threaded by design, so the adapter pins
search to one thread. vecdb's single-thread numbers are its strongest
(matches FAISS recall, faster QPS on SIFT1M on Apple Silicon); on the x86 CI
hardware results will reflect the AVX2 build path.
