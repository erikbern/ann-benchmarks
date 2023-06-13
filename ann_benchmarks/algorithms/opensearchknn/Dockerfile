# Warning! Do not use this config in production! 
# This is only for testing and security has been turned off.

FROM ann-benchmarks AS builder

ARG OPENSEARCH_VERSION=2.6.0

WORKDIR /tmp

# Download OpenSearch to intermediate builder.
RUN apt-get -y install curl

RUN set -eux ; \
    ARCH=''; \
    case "$(arch)" in \
        aarch64) ARCH='arm64' ;; \
        x86_64)  ARCH='x64' ;; \
        *) echo >&2 ; echo >&2 "Unsupported architecture $(arch)" ; echo >&2 ; exit 1 ;; \
    esac ; \
    curl -OsS https://artifacts.opensearch.org/releases/bundle/opensearch/${OPENSEARCH_VERSION}/opensearch-${OPENSEARCH_VERSION}-linux-$ARCH.tar.gz; \
    tar -zxf opensearch-${OPENSEARCH_VERSION}-linux-$ARCH.tar.gz; \
    mv opensearch-${OPENSEARCH_VERSION} opensearch; \
    rm -r opensearch-${OPENSEARCH_VERSION}-linux-$ARCH.tar.gz opensearch/plugins/opensearch-security opensearch/plugins/opensearch-alerting opensearch/plugins/opensearch-anomaly-detection opensearch/plugins/opensearch-asynchronous-search opensearch/plugins/opensearch-cross-cluster-replication opensearch/plugins/opensearch-geospatial opensearch/plugins/opensearch-index-management opensearch/plugins/opensearch-job-scheduler opensearch/plugins/opensearch-ml opensearch/plugins/opensearch-neural-search opensearch/plugins/opensearch-notifications opensearch/plugins/opensearch-notifications-core opensearch/plugins/opensearch-observability opensearch/plugins/opensearch-performance-analyzer opensearch/plugins/opensearch-reports-scheduler opensearch/plugins/opensearch-security-analytics opensearch/plugins/opensearch-sql;

# Install OpenSearch in final image:
#  - https://opensearch.org/docs/latest/install-and-configure/install-opensearch/tar/
FROM ann-benchmarks

WORKDIR /home/app

RUN apt-get -y install curl tmux gosu

COPY --from=builder /tmp/opensearch /home/app/opensearch
RUN chmod -R 777 opensearch

RUN echo "vm.max_map_count=262144" >> /etc/sysctl.conf

RUN python3 -m pip install --upgrade opensearch-py==2.2.0 tqdm

# Configure OpenSearch for single-node, single-core.
RUN echo '\
discovery.type: single-node\n\
network.host: 0.0.0.0\n\
node.roles: [ data, master ]\n\
node.processors: 1\n\
thread_pool.write.size: 1\n\
thread_pool.search.size: 1\n\
thread_pool.search.queue_size: 1' > opensearch/config/opensearch.yml

RUN echo '\
-Xms3G\n\
-Xmx3G\n\
-XX:InitiatingHeapOccupancyPercent=80\n\
-XX:+HeapDumpOnOutOfMemoryError\n\
-XX:HeapDumpPath=data\n\
-XX:ErrorFile=logs/hs_err_pid%p.log\n\
-Xlog:gc*,gc+age=trace,safepoint:file=logs/gc.log:utctime,pid,tags:filecount=32,filesize=64m' > opensearch/config/jvm.options

# Custom entrypoint that also starts the OpenSearch server.
RUN useradd -m opensearch
RUN echo 'tmux new-session -d -s opensearch """exec gosu opensearch "./opensearch/opensearch-tar-install.sh""""' > entrypoint.sh
RUN echo 'python3 -u run_algorithm.py "$@"' >> entrypoint.sh

ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
