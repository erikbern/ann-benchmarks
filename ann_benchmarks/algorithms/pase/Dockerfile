FROM ann-benchmarks

# Set timezone to prevent interactive prompts
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install PostgreSQL build dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    libxml2-dev \
    libxslt-dev \
    libssl-dev \
    libxml2-utils \
    xsltproc \
    ccache \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and build PostgreSQL 11.0
WORKDIR /tmp
RUN wget https://ftp.postgresql.org/pub/source/v11.0/postgresql-11.0.tar.gz && \
    tar -zxvf postgresql-11.0.tar.gz && \
    cd postgresql-11.0 && \
    ./configure --prefix=/usr/local/pgsql CFLAGS="-O3" LDFLAGS="-fPIC -fopenmp" && \
    make -j$(nproc) && \
    make install

# Clone VecDB and copy PASE files
RUN git clone https://github.com/YunanZzz/VecDB-ICDE24.git /vecdb && \
    mkdir -p /tmp/postgresql-11.0/contrib/pase && \
    cp -r /vecdb/postgresql-11.0/contrib/pase/* /tmp/postgresql-11.0/contrib/pase/

# Build and install PASE
WORKDIR /tmp/postgresql-11.0/contrib/pase
RUN PG_CONFIG=/usr/local/pgsql/bin/pg_config make USE_PGXS=1 && \
    make install

# Create postgres user and set up directories with proper permissions
RUN useradd postgres && \
    mkdir -p /tmp /var/run/postgresql /usr/local/pgsql/data /usr/local/pgsql/logs && \
    chown -R postgres:postgres /tmp && \
    chown -R postgres:postgres /var/run/postgresql && \
    chown -R postgres:postgres /usr/local/pgsql && \
    chmod 0700 /usr/local/pgsql/data && \
    chmod 0700 /usr/local/pgsql/logs && \
    su postgres -c '/usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data'

# Configure PostgreSQL
USER postgres
RUN echo "unix_socket_directories = '/tmp'" >> /usr/local/pgsql/data/postgresql.conf && \
    echo "shared_buffers = 4GB" >> /usr/local/pgsql/data/postgresql.conf && \
    echo "listen_addresses = '*'" >> /usr/local/pgsql/data/postgresql.conf && \
    echo "local   all             all                                     trust" > /usr/local/pgsql/data/pg_hba.conf && \
    echo "host    all             all             127.0.0.1/32            trust" >> /usr/local/pgsql/data/pg_hba.conf && \
    echo "host    all             all             ::1/128                 trust" >> /usr/local/pgsql/data/pg_hba.conf && \
    echo "host    all             all             0.0.0.0/0               trust" >> /usr/local/pgsql/data/pg_hba.conf && \
    /usr/local/pgsql/bin/pg_ctl start -D /usr/local/pgsql/data -l /usr/local/pgsql/logs/logfile -w && \
    /usr/local/pgsql/bin/createuser -s ann && \
    /usr/local/pgsql/bin/createdb -O ann ann && \
    /usr/local/pgsql/bin/pg_ctl stop -D /usr/local/pgsql/data

USER root
# Expose PostgreSQL port

EXPOSE 5432

# Create logging configuration
COPY <<-'EOF' /home/app/logging.conf
[loggers]
keys=root

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=
EOF

# Add PostgreSQL binaries to PATH
ENV PATH=/usr/local/pgsql/bin:$PATH

# Install Python PostgreSQL adapter and vector support
RUN pip install psycopg[binary]

WORKDIR /home/app

# Set permissions for app directory after copying
RUN mkdir -p /home/app/ann_benchmarks/algorithms/pase && \
    chown -R postgres:postgres /home/app && \
    chmod -R 0700 /home/app