FROM cassandra:5.0.4

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir cassandra-driver

COPY run_algorithm.py .
COPY ann_benchmarks/algorithms/cassandra/entrypoint.sh ann_benchmarks/algorithms/cassandra/usr/local/bin/entrypoint.sh
RUN chmod +x ann_benchmarks/algorithms/cassandra/usr/local/bin/entrypoint.sh

EXPOSE 9042

ENTRYPOINT ["entrypoint.sh"]