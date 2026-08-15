FROM ann-benchmarks

ARG HANN_REF=v0.8.0
ARG GO_VERSION=1.26.5

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /home/app/hann_bridge
RUN git clone https://github.com/habedi/hann hann-src && cd hann-src && git checkout "${HANN_REF}"
COPY ann_benchmarks/algorithms/hann/bridge.go ann_benchmarks/algorithms/hann/go.mod ./
RUN go mod tidy && CGO_ENABLED=1 go build -buildmode=c-shared -o /home/app/libhann_bridge.so .

WORKDIR /home/app
COPY ann_benchmarks/__init__.py ann_benchmarks/
COPY ann_benchmarks/algorithms/__init__.py ann_benchmarks/algorithms/
COPY ann_benchmarks/algorithms/base/module.py ann_benchmarks/algorithms/base/
COPY ann_benchmarks/algorithms/hann/module.py ann_benchmarks/algorithms/hann/
RUN python -c 'import ctypes; from ann_benchmarks.algorithms.hann.module import Hann; h = Hann("angular", {"M": 16, "efConstruction": 200}); import numpy as np; X = np.random.rand(100, 8).astype(np.float32); h.fit(X); h.set_query_arguments(10); r = h.query(X[0], 5); assert r[0] == 0, r; h.done(); print("hann smoke check passed:", r)' \
    && rm -rf ann_benchmarks
