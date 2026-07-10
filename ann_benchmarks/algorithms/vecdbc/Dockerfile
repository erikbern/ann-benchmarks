FROM ann-benchmarks

# vecdb is a C library compiled at install time; needs a C toolchain with
# OpenMP. The sdist picks portable SIMD flags per arch (AVX2 on x86).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir vecdbc==0.1.1

# sanity check: import must succeed at build time
RUN python3 -c "from pyvecdb import VecDB; import numpy as np; \
    d=VecDB(dim=8); d.add(np.random.rand(50,8).astype(np.float32)); \
    print('vecdbc import OK', len(d))"
