FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

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

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "GRANT ALL ON SCHEMA public TO ann" && \
    psql -d ann -c "CREATE EXTENSION vector" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '4GB'" && \
    psql -c "ALTER USER ann SET max_parallel_maintenance_workers = 0" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '4GB'"
USER root

RUN pip install psycopg[binary] pgvector
