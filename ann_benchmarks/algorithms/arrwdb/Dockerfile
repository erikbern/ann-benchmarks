FROM ann-benchmarks

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin
RUN pip install maturin numpy

# Clone and build arrwDB's Rust HNSW index
RUN git clone https://github.com/bledden/arrwDB.git /opt/arrwdb
RUN cd /opt/arrwdb/rust/indexes && \
    RUSTFLAGS='-C target-cpu=native' maturin build --release && \
    pip install target/wheels/*.whl

RUN python -c 'from rust_hnsw import RustFastHNSWIndex; print("arrwDB OK")'
