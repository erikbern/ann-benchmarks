FROM ann-benchmarks

# Python layer from PyPI; the native libraries are rebuilt below.
RUN pip install quantaldb==0.1.2

# quantal fixes the vector dimension at compile time and compiles missing
# dimensions on demand from a source checkout. Zig is distributed as a
# single static tarball; no system toolchain needed.
ARG ZIG_VERSION=0.16.0
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt \
    && ln -s /opt/zig-x86_64-linux-${ZIG_VERSION}/zig /usr/local/bin/zig
RUN git clone --branch v0.1.2 --depth 1 https://github.com/PaytonWebber/quantal /opt/quantal
ENV QUANTAL_SRC=/opt/quantal

# The wheel bundles portable baseline-x86-64 libraries. Benchmark numbers
# must come from native builds (the kernels are SIMD-heavy), so drop the
# bundled libraries and precompile every benchmark-dataset dimension on
# this machine instead.
RUN python -c "import quantal, os, shutil; \
    shutil.rmtree(os.path.join(os.path.dirname(quantal.__file__), '_libs'), ignore_errors=True)"
RUN python -c "from quantal import Index; \
    [Index(dim=d) for d in (16, 20, 25, 50, 64, 96, 100, 200, 256, 512, 1536)]"

RUN python -c "from quantal import Index; Index(dim=1536)"
