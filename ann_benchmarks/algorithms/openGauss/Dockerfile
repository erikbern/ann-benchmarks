FROM ann-benchmarks

RUN add-apt-repository ppa:opengauss/opengauss

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN sed -i 's/bookworm/jammy/g' /etc/apt/sources.list.d/opengauss-ubuntu-opengauss-bookworm.list
RUN apt-get update && apt-get install -y --no-install-recommends build-essential 
RUN apt install opengauss
RUN sh -c 'echo "local all all trust" > /var/lib/opengauss/data/pg_hba.conf'

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

USER opengauss
RUN service opengauss start && \
    gsql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    gsql -c "CREATE DATABASE ann" && \
    gsql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    gsql -d ann -c "GRANT ALL ON SCHEMA public TO ann" && \
    gsql -c "ALTER USER ann SET maintenance_work_mem = '4GB'" && \
    gsql -c "ALTER USER ann SET max_parallel_maintenance_workers = 0" && \
    gsql -c "ALTER SYSTEM SET shared_buffers = '4GB'"
USER root

RUN pip install psycopg[binary] 