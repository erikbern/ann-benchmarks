FROM ann-benchmarks

COPY build_bench/tests/bench/ann-benchmark/pyiresearch*.so \
     /usr/local/lib/python3.12/dist-packages/

RUN python3 -c 'import pyiresearch; print("pyiresearch OK")'
