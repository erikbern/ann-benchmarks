FROM ann-benchmarks

RUN apt-get update && apt-get install -y wget xz-utils git && \
    rm -rf /var/lib/apt/lists/*

# Install Zig 0.15.2 (stable)
RUN wget -q https://ziglang.org/download/0.15.2/zig-x86_64-linux-0.15.2.tar.xz && \
    tar xf zig-x86_64-linux-0.15.2.tar.xz && \
    mv zig-x86_64-linux-0.15.2 /opt/zig && \
    rm zig-x86_64-linux-0.15.2.tar.xz
ENV PATH="/opt/zig:${PATH}"

# Clone and build natively on x86_64
RUN git clone https://github.com/mdvsh/hnsw-zig.git /opt/hnsw-zig
WORKDIR /opt/hnsw-zig
RUN zig build -Doptimize=ReleaseFast

# Install shared library
RUN mkdir -p /home/app/ann_benchmarks/algorithms/hnswzig/lib && \
    cp zig-out/lib/libhnsw_zig.so /home/app/ann_benchmarks/algorithms/hnswzig/lib/

# Copy the full ann_benchmarks package (install.py normally does this)
COPY ann_benchmarks/ /home/app/ann_benchmarks/

# Verify .so loads
RUN ldd /home/app/ann_benchmarks/algorithms/hnswzig/lib/libhnsw_zig.so && \
    python3 -c "import ctypes; ctypes.CDLL('/home/app/ann_benchmarks/algorithms/hnswzig/lib/libhnsw_zig.so'); print('OK')"

# Reset workdir so ann-benchmarks harness works
WORKDIR /home/app
