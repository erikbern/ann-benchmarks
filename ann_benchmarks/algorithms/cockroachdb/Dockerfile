ARG COCKROACHDB_VERSION=latest
FROM cockroachdb/cockroach:${COCKROACHDB_VERSION} AS cockroach

FROM ann-benchmarks

# Copy cockroach binary from official image
COPY --from=cockroach /cockroach/cockroach /usr/local/bin/cockroach

# Python dependencies
RUN pip install psycopg[binary] pgvector

# CockroachDB data directory
RUN mkdir -p /tmp/cockroach-data

# Custom entrypoint: start CockroachDB then run benchmark
RUN printf '#!/bin/bash\n\
set -eu\n\
cockroach start-single-node --insecure \
  --store=/tmp/cockroach-data \
  --listen-addr=localhost:26257 \
  --http-addr=localhost:8080 \
  --background\n\
for i in $(seq 1 30); do\n\
  cockroach sql --insecure -e "SELECT 1" 2>/dev/null && break\n\
  sleep 1\n\
done\n\
python3 -u run_algorithm.py "$@"\n' > /home/app/entrypoint.sh && \
    chmod +x /home/app/entrypoint.sh

ENTRYPOINT ["/home/app/entrypoint.sh"]
