FROM ann-benchmarks AS builder
ARG ELASTICSEARCH_VERSION=8.7.0

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y curl

# Download Elasticsearch to intermediate builder.
WORKDIR /tmp
RUN curl -OsS https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ELASTICSEARCH_VERSION}-linux-$(arch).tar.gz
RUN curl -sS https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ELASTICSEARCH_VERSION}-linux-$(arch).tar.gz.sha512 | shasum -a 512 -c -

WORKDIR /usr/share/elasticsearch
RUN tar -zxf /tmp/elasticsearch-${ELASTICSEARCH_VERSION}-linux-$(arch).tar.gz --strip-components=1

# Install Elasticsearch in final image:
#  - https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html
#  - https://www.elastic.co/guide/en/elasticsearch/reference/current/system-config.html
FROM ann-benchmarks
ARG ELASTICSEARCH_VERSION=8.7.0

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y curl

WORKDIR /usr/share/elasticsearch

# Create elasticsearch user and user group.
RUN groupadd -g 1000 elasticsearch
RUN adduser --uid 1000 --gid 1000 --home /usr/share/elasticsearch elasticsearch

COPY --from=builder --chown=elasticsearch:elasticsearch /usr/share/elasticsearch /usr/share/elasticsearch

RUN echo "vm.max_map_count=262144" >> /etc/sysctl.conf

# Backup original configurations for potential future reference.
RUN cp config/elasticsearch.yml config/elasticsearch.yml.bak
RUN cp config/jvm.options config/jvm.options.bak

# Configure Elasticsearch for single-node, single-core.
RUN echo '\
discovery.type: single-node\n\
node.roles: [master, data]\n\
node.processors: 1\n\
path.data: /usr/share/elasticsearch/data\n\
path.logs: /usr/share/elasticsearch/logs\n\
bootstrap.memory_lock: true\n\
thread_pool.write.size: 1\n\
thread_pool.search.size: 1\n\
thread_pool.search.queue_size: 1\n\
xpack.security.enabled: false\n\
' > config/elasticsearch.yml

RUN echo '\
-Xms3G\n\
-Xmx3G\n\
-XX:+UseG1GC\n\
-XX:HeapDumpPath=data\n\
-XX:ErrorFile=/usr/share/elasticsearch/logs/hs_err_pid%p.log\n\
-Xlog:gc*,gc+age=trace,safepoint:file=/usr/share/elasticsearch/logs/gc.log:utctime,pid,tags:filecount=32,filesize=64m\n\
' > config/jvm.options

RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch

WORKDIR /home/app

RUN python3 -m pip install elasticsearch==${ELASTICSEARCH_VERSION}

# Custom entrypoint that also starts the Elasticsearch server.
RUN echo 'set -eux' >> entrypoint.sh
RUN echo 'su - elasticsearch -c "nohup /usr/share/elasticsearch/bin/elasticsearch > nohup.out 2>&1 &"' >> entrypoint.sh
RUN echo 'python3 -u run_algorithm.py "$@"' >> entrypoint.sh

ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
