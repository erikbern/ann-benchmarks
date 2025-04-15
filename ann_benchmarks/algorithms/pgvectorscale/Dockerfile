FROM ann-benchmarks

RUN git clone https://github.com/pgvector/pgvector /tmp/pgvector

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql-common
RUN /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
RUN apt-get install -y --no-install-recommends postgresql-16 postgresql-server-dev-16
RUN sh -c 'echo "local all all trust" > /etc/postgresql/16/main/pg_hba.conf'

# Dynamically set OPTFLAGS based on the architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        OPTFLAGS="-march=native -msve-vector-bits=512"; \
    elif [ "$ARCH" = "x86_64" ]; then \
        OPTFLAGS="-march=native -mprefer-vector-width=512"; \
    else \
        OPTFLAGS="-march=native"; \
    fi && \
    cd /tmp/pgvector && \
    make clean && \
    make OPTFLAGS="$OPTFLAGS" && \
    make install

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Ensure the cargo bin directory is in the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pgrx using cargo
RUN cargo install --locked cargo-pgrx@0.12.9
# Initialize pgrx with the PostgreSQL version
RUN cargo pgrx init --pg16 pg_config
RUN git clone https://github.com/timescale/pgvectorscale /tmp/pgvectorscale
RUN cd /tmp/pgvectorscale/pgvectorscale && \
    cargo pgrx install --release

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "GRANT ALL ON SCHEMA public TO ann" && \
    psql -d ann -c "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '4GB'" && \
    psql -c "ALTER USER ann SET max_parallel_maintenance_workers = 0" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '4GB'"
USER root

RUN pip install psycopg[binary] pgvector
