FROM ann-benchmarks AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Rust toolchain
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.98.1
ENV PATH="/root/.cargo/bin:${PATH}"

# maturin builds the PyO3 extension
RUN pip install maturin

ARG ANNEX_REPO=https://github.com/rosharma719/ANNex.git
ARG ANNEX_REF=d963df612025f9cbc7a2811eb29595e3c55e8448

RUN git init /annex \
    && cd /annex \
    && git remote add origin "$ANNEX_REPO" \
    && git fetch --depth 1 origin "$ANNEX_REF" \
    && git checkout --detach FETCH_HEAD

RUN cd /annex/python/annex-py \
    && maturin build --release --features extension-module -o /tmp/annex-wheels \
    && find /annex/target -mindepth 1 -delete \
    && find /root/.cargo/registry -mindepth 1 -delete

FROM ann-benchmarks

COPY --from=builder /tmp/annex-wheels /tmp/annex-wheels
RUN pip install /tmp/annex-wheels/*.whl \
    && find /tmp/annex-wheels -mindepth 1 -delete
