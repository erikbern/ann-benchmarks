FROM ann-benchmarks

RUN apt-get -y install curl

# Install RUST
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install SurrealDB 
RUN curl --proto '=https' --tlsv1.2 -sSf https://install.surrealdb.com | sh -s -- --nightly

# Install the Python client
RUN apt-get install -y python3-dev -y
RUN apt-get install -y python3-pip -y
RUN pip install --upgrade pip setuptools wheel
RUN apt-get install libclang-dev -y
RUN pip install git+https://github.com/surrealdb/surrealdb.py@rust-no-runtime