FROM ann-benchmarks

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    curl \
    git \
    libclang-dev \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/sdslabs/VortexDB.git /opt/vortexdb
RUN cd /opt/vortexdb && cargo build --release -p server
RUN pip3 install /opt/vortexdb/client/python

RUN python3 -c 'import vortexdb'
