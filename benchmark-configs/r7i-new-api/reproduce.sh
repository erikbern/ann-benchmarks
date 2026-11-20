#!/usr/bin/env bash
# Run from the c7ibench checkout. Use frozen source checkouts and a fresh output.
set -euo pipefail
: "${BENCH_ROOT:?Path containing the full HDF5 datasets and fingerprinted historical base indexes}"
: "${SUCCINCT_SOURCE:?Path to the succinct source checkout}"
: "${EXPLORATION_SOURCE:?Path to the exploration source checkout}"
: "${ARTIFACT_ROOT:?New output directory; never reuse an existing measurement directory}"
QG_PYTHON="${QG_PYTHON:-$BENCH_ROOT/.venv/bin/python}"
export PATH="$BENCH_ROOT/.venv/bin:$PATH"
# On the original machine numactl was unpacked locally, without root installation.
if ! command -v numactl >/dev/null; then
  export PATH="$BENCH_ROOT/results/r7i-new-api-20261119/local/usr/bin:$PATH"
fi
command -v numactl >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QG_THP_POLICY=prepare
unset QG_THP_MIN_COVERAGE
test ! -e "$ARTIFACT_ROOT"
mkdir -p "$ARTIFACT_ROOT"
for variant in succinct exploration; do
  if [ "$variant" = succinct ]; then
    QG_SOURCE="$SUCCINCT_SOURCE"
    QG_COMMIT=21c259867371586726e772145dcfa4b701af2d5f
  else
    QG_SOURCE="$EXPLORATION_SOURCE"
    QG_COMMIT=35afef03b24c5e8fa452b1f1523142471a53afbe
  fi
  # Documentation/result commits may follow the measured revision; execution code must match.
  git -C "$QG_SOURCE" diff --exit-code "$QG_COMMIT" -- src cmake CMakeLists.txt \
    tools/core/qg_bench.cc tools/core/CMakeLists.txt tools/core/bench/qg_bench.py
  git -C "$QG_SOURCE" submodule update --init --recursive
  cmake -S "$QG_SOURCE" -B "$QG_SOURCE/build-r7i" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON \
    -DBUILD_TOOLS=ON -DBUILD_C_BINDINGS=OFF -DBUILD_ZVEC_SHARED=OFF \
    -DBUILD_ZVEC_CORE_SHARED=OFF -DBUILD_ZVEC_AILEGO_SHARED=OFF -DENABLE_ICELAKE=ON
  cmake --build "$QG_SOURCE/build-r7i" --target qg_bench quantized_graph_index_test -j 12
  "$QG_SOURCE/build-r7i/bin/quantized_graph_index_test"
  "$QG_PYTHON" "$QG_SOURCE/tools/core/bench/qg_preflight.py" \
    --binary "$QG_SOURCE/build-r7i/bin/qg_bench" \
    --config-dir benchmark-configs/r7i-new-api --output "$ARTIFACT_ROOT/preflight-$variant"
  for phase in check prepare build; do
    "$QG_PYTHON" scripts/run_native_qg.py "$phase" --zvec-source "$QG_SOURCE" \
      --bench-root "$BENCH_ROOT" --artifacts "$ARTIFACT_ROOT/$variant"
  done
done
"$QG_PYTHON" "$SUCCINCT_SOURCE/tools/core/bench/qg_bench.py" isolate \
  --manifest "$ARTIFACT_ROOT/isolation.json" -- \
  "$QG_PYTHON" scripts/measure_native_qg_variants.py \
    --succinct "$SUCCINCT_SOURCE" --exploration "$EXPLORATION_SOURCE" \
    --bench-root "$BENCH_ROOT" --artifacts "$ARTIFACT_ROOT"
"$QG_PYTHON" scripts/report_native_qg.py --artifacts "$ARTIFACT_ROOT" \
  --bench-root "$BENCH_ROOT" \
  --history "benchmark-configs/r7i-new-api/succinct/history/20261118-845fe9ec"
