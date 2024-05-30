FROM ann-benchmarks

RUN git clone https://github.com/neondatabase/pg_embedding /tmp/pg_embedding

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql postgresql-server-dev-all
RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/pg_embedding && \
	make clean && \
	make OPTFLAGS="-march=native -mprefer-vector-width=512" && \
	make install

# Modify shared_buffers in postgresql.conf to 16GB
# RUN sed -i 's/#shared_buffers = 128MB/shared_buffers = 16GB/' /etc/postgresql/14/main/postgresql.conf

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION embedding" && \
    psql -d ann -c "CREATE EXTENSION pg_prewarm"
USER root

RUN pip install psycopg[binary]
