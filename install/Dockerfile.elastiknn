FROM ann-benchmarks

WORKDIR /home/app

# Install elasticsearch.
ENV DEBIAN_FRONTEND noninteractive
RUN apt install -y wget curl htop
RUN wget --quiet https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-amd64.deb \
    && dpkg -i elasticsearch-oss-7.9.2-amd64.deb \
    && rm elasticsearch-oss-7.9.2-amd64.deb

# Install python client.
RUN python3 -m pip install --upgrade elastiknn-client==0.1.0rc47

# Install plugin.
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch \
    https://github.com/alexklibisz/elastiknn/releases/download/0.1.0-PRE47/elastiknn-0.1.0-PRE47_es7.9.2.zip

# Configure elasticsearch and JVM for single-node, single-core.
RUN cp /etc/elasticsearch/jvm.options /etc/elasticsearch/jvm.options.bak
RUN cp /etc/elasticsearch/elasticsearch.yml /etc/elasticsearch/elasticsearch.yml.bak

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
-Xlog:gc*,gc+age=trace,safepoint:file=/var/log/elasticsearch/gc.log:utctime,pid,tags:filecount=32,filesize=64m\n\
-Dcom.sun.management.jmxremote.ssl=false\n\
-Dcom.sun.management.jmxremote.authenticate=false\n\
-Dcom.sun.management.jmxremote.local.only=false\n\
-Dcom.sun.management.jmxremote.port=8097\n\
-Dcom.sun.management.jmxremote.rmi.port=8097\n\
-Djava.rmi.server.hostname=localhost' > /etc/elasticsearch/jvm.options

# JMX port. Need to also map the port when running.
EXPOSE 8097

# Make sure you can start the service.
RUN service elasticsearch start && service elasticsearch stop

# Custom entrypoint that also starts the Elasticsearch server.\
RUN echo 'service elasticsearch start && python3 -u run_algorithm.py "$@"' > entrypoint.sh
ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
