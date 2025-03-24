FROM ann-benchmarks

# Install curl.
ENV DEBIAN_FRONTEND noninteractive
RUN apt install -y curl

# Create an app user, as elasticsearch cannot run as root.
RUN mkdir -p /home/elasticsearch
RUN groupadd elasticsearch
RUN useradd elasticsearch -g elasticsearch -p elasticsearch
RUN chown -R elasticsearch:elasticsearch /home/elasticsearch
WORKDIR /home/elasticsearch
USER elasticsearch

# Install elasticsearch.
RUN curl -o elasticsearch.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz
RUN tar xzf elasticsearch.tar.gz
RUN mv elasticsearch-* elasticsearch && rm elasticsearch.tar.gz

# Install plugin.
RUN /home/elasticsearch/elasticsearch/bin/elasticsearch-plugin install --batch \
    https://github.com/alexklibisz/elastiknn/releases/download/8.15.0.1/elastiknn-8.15.0.1.zip

# Configuration
# Backup the original configurations, which can be useful for comparing.
RUN cp elasticsearch/config/jvm.options elasticsearch/config/jvm.options.bak
RUN cp elasticsearch/config/elasticsearch.yml elasticsearch/config/elasticsearch.yml.bak

# Configure elasticsearch and JVM for single-node, single-core.
RUN echo '\
cluster.initial_master_nodes: main\n\
node.name: main\n\
node.roles: [master,data]\n\
node.processors: 1\n\
path.data: /home/elasticsearch/elasticsearch/data\n\
path.logs: /home/elasticsearch/elasticsearch/logs\n\
thread_pool.write.size: 1\n\
thread_pool.search.size: 1\n\
thread_pool.search.queue_size: 1\n\
xpack.security.enabled: false\n\
' > elasticsearch/config/elasticsearch.yml

RUN echo '\
-Xms3G\n\
-Xmx3G\n\
-XX:+UseG1GC\n\
-XX:G1ReservePercent=25\n\
-XX:InitiatingHeapOccupancyPercent=30\n\
-XX:+HeapDumpOnOutOfMemoryError\n\
-XX:HeapDumpPath=/home/elasticsearch/elasticsearch/logs\n\
-XX:ErrorFile=/home/elasticsearch/elasticsearch/logs/hs_err_pid%p.log\n\
-Xlog:gc*,gc+age=trace,safepoint:file=/home/elasticsearch/elasticsearch/logs/gc.log:utctime,pid,tags:filecount=32,filesize=64m\n\
-Dcom.sun.management.jmxremote.ssl=false\n\
-Dcom.sun.management.jmxremote.authenticate=false\n\
-Dcom.sun.management.jmxremote.local.only=false\n\
-Dcom.sun.management.jmxremote.port=8097\n\
-Dcom.sun.management.jmxremote.rmi.port=8097\n\
-Djava.rmi.server.hostname=localhost' > elasticsearch/config/jvm.options


# JMX port. Need to also map the port when running.
EXPOSE 8097

# Switch back to root because that's what CI expects.
USER root
WORKDIR /home/app

# Install python client.
RUN python3 -m pip install elastiknn-client==8.15.0.1

# Custom entrypoint that also starts the Elasticsearch server in the background
RUN echo '\
set -e\n\
su - elasticsearch -c "nohup /home/elasticsearch/elasticsearch/bin/elasticsearch > nohup.out &"\n\
python3 -u run_algorithm.py "$@" || (cat nohup.out && exit 1)\n\
' > /home/app/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
