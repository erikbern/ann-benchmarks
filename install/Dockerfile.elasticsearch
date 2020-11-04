FROM ann-benchmarks

WORKDIR /home/app

# Install elasticsearch.
ENV DEBIAN_FRONTEND noninteractive
RUN apt install -y wget curl htop
RUN wget --quiet https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-amd64.deb \
    && dpkg -i elasticsearch-7.9.2-amd64.deb \
    && rm elasticsearch-7.9.2-amd64.deb

# Install python client.
RUN python3 -m pip install --upgrade elasticsearch==7.9.1

# Configure elasticsearch and JVM for single-node, single-core.
RUN echo '\
discovery.type: single-node\n\
network.host: 0.0.0.0\n\
node.master: true\n\
node.data: true\n\
node.processors: 1\n\
thread_pool.write.size: 1\n\
thread_pool.search.size: 1\n\
thread_pool.search.queue_size: 1\n\
path.data: /var/lib/elasticsearch\n\
path.logs: /var/log/elasticsearch\n\
' > /etc/elasticsearch/elasticsearch.yml

RUN echo '\
-Xms3G\n\
-Xmx3G\n\
-XX:+UseG1GC\n\
-XX:G1ReservePercent=25\n\
-XX:InitiatingHeapOccupancyPercent=30\n\
-XX:+HeapDumpOnOutOfMemoryError\n\
-XX:HeapDumpPath=/var/lib/elasticsearch\n\
-XX:ErrorFile=/var/log/elasticsearch/hs_err_pid%p.log\n\
-Xlog:gc*,gc+age=trace,safepoint:file=/var/log/elasticsearch/gc.log:utctime,pid,tags:filecount=32,filesize=64m' > /etc/elasticsearch/jvm.options

# Make sure you can start the service.
RUN service elasticsearch start && service elasticsearch stop

# Custom entrypoint that also starts the Elasticsearch server.
RUN echo 'service elasticsearch start && python3 -u run_algorithm.py "$@"' > entrypoint.sh
ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
