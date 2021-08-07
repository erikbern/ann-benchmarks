# Warning! Do not use this config in production! 
# This is only for testing and security has been turned off.

FROM ann-benchmarks

WORKDIR /home/app

# Install Opensearch following instructions from https://opensearch.org/docs/opensearch/install/tar/
# and https://opensearch.org/docs/opensearch/install/important-settings/
RUN apt-get install tmux wget gosu -y
RUN wget https://artifacts.opensearch.org/releases/bundle/opensearch/1.0.0/opensearch-1.0.0-linux-x64.tar.gz
RUN tar -zxf opensearch-1.0.0-linux-x64.tar.gz
RUN rm -r opensearch-1.0.0-linux-x64.tar.gz opensearch-1.0.0/plugins/opensearch-security
RUN chmod -R 777 opensearch-1.0.0
RUN sysctl -w vm.max_map_count=262144

# Install python client.
RUN python3 -m pip install --upgrade elasticsearch==7.13.4 tqdm

# Configure opensearch for single-node, single-core.
RUN echo '\
discovery.type: single-node\n\
network.host: 0.0.0.0\n\
node.roles: [ data, master ]\n\
node.processors: 1\n\
thread_pool.write.size: 1\n\
thread_pool.search.size: 1\n\
thread_pool.search.queue_size: 1' > opensearch-1.0.0/config/opensearch.yml

RUN echo '\
-Xms3G\n\
-Xmx3G\n\
-XX:InitiatingHeapOccupancyPercent=30\n\
-XX:+HeapDumpOnOutOfMemoryError\n\
-XX:HeapDumpPath=data\n\
-XX:ErrorFile=logs/hs_err_pid%p.log\n\
-Xlog:gc*,gc+age=trace,safepoint:file=logs/gc.log:utctime,pid,tags:filecount=32,filesize=64m' > opensearch-1.0.0/config/jvm.options

# Custom entrypoint that also starts the Opensearch server
RUN useradd -m opensearch
RUN echo 'tmux new-session -d -s opensearch """exec gosu opensearch "./opensearch-1.0.0/opensearch-tar-install.sh""""' > entrypoint.sh
RUN echo 'python3 -u run_algorithm.py "$@"' >> entrypoint.sh
ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]