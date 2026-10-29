# Yellow Phoenix engine image for ann-benchmarks.
# Built from the public yellow-phoenix-release source with the designated
# PUBLIC build phrase (internal/paper builds use a different phrase; the
# public one is baked here for reproducibility — no secret in the image).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV YP_GENESIS_PHRASE="yellow-phoenix-ann-benchmarks-2026"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-numpy python3-pip python3-h5py python3-psutil \
    curl git build-essential pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.1
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone --depth 1 https://github.com/Alfredai2025/yellow-phoenix-release.git /src
WORKDIR /src
RUN cargo build --release --bin build_hnsw_from_flat_ism --bin yp_ann_server \
    && mkdir -p /usr/local/yp/bin \
    && cp target/release/build_hnsw_from_flat_ism target/release/yp_ann_server /usr/local/yp/bin/

ENV YP_ENGINE=/usr/local/yp/bin
