DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
apt-get update && apt-get install -y --no-install-recommends build-essential postgresql-common
/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
apt-get install -y --no-install-recommends postgresql-16 postgresql-server-dev-16

ARCH=$(uname -m) && \
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

pip install psycopg[binary] pgvector

cd .. && [ ! -d "pgvector-gpu-python" ] && git clone https://github.com/YOUR_USERNAME/pgvector-gpu-python.git; 
cd pgvector-gpu-python && pip install -e . --quiet

sudo sed -i 's/local.*postgres.*peer/local all postgres trust/' /etc/postgresql/16/main/pg_hba.conf
