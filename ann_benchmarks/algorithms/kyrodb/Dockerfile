ARG ANNB_BASE=ann-benchmarks:latest
ARG RUST_VERSION=1.93.1
FROM rust:${RUST_VERSION}-bookworm AS builder

ARG TARGETARCH
ARG KYRODB_GIT=https://github.com/KyroDB/KyroDB.git
ARG KYRODB_REF=7f113a73c6d6853a2c111e78dc87ed53acc95bee
ARG RUST_CPU_AMD64=x86-64-v3
ARG RUST_CPU_ARM64=generic
ARG RUSTFLAGS_EXTRA=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN git clone "$KYRODB_GIT" /src/kyrodb \
    && cd /src/kyrodb \
    && git fetch --tags --prune --no-recurse-submodules origin \
    && git checkout --detach "$KYRODB_REF"

WORKDIR /src/kyrodb
RUN case "${TARGETARCH}" in \
        amd64) export KYRO_RUST_CPU="${RUST_CPU_AMD64}" ;; \
        arm64) export KYRO_RUST_CPU="${RUST_CPU_ARM64}" ;; \
        *) export KYRO_RUST_CPU=generic ;; \
    esac \
    && export RUSTFLAGS="-C target-cpu=${KYRO_RUST_CPU} ${RUSTFLAGS_EXTRA}" \
    && cargo build --release --locked -p kyrodb-engine --lib --features ffi-bench

FROM ${ANNB_BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY ann_benchmarks/algorithms/kyrodb /home/app/ann_benchmarks/algorithms/kyrodb
COPY --from=builder /src/kyrodb/target/release/libkyrodb_engine.so /usr/local/lib/libkyrodb_engine.so

ENV LD_LIBRARY_PATH=/usr/local/lib \
    RAYON_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1
