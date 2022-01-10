# Install Milvus
FROM milvusdb/milvus:0.6.0-cpu-d120719-2b40dd as milvus
RUN apt-get update && apt-get install -y wget
RUN wget https://raw.githubusercontent.com/milvus-io/docs/master/v0.6.0/assets/server_config.yaml
RUN sed -i 's/cpu_cache_capacity: 16/cpu_cache_capacity: 4/' server_config.yaml  # otherwise my Docker blows up
RUN mv server_config.yaml /var/lib/milvus/conf/server_config.yaml

# Switch back to ANN-benchmarks base image and copy all files
FROM ann-benchmarks
COPY --from=milvus /var/lib/milvus /var/lib/milvus
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/var/lib/milvus/lib"
RUN apt-get update
RUN apt-get install -y libmysqlclient-dev

# Python client
RUN pip3 install pymilvus==0.2.7

# Fixing some version incompatibility thing
RUN pip3 install numpy==1.18 scipy==1.1.0 scikit-learn==0.21

# Dumb entrypoint thing that runs the daemon as well
RUN echo '#!/bin/bash' >> entrypoint.sh
RUN echo '/var/lib/milvus/bin/milvus_server -d -c /var/lib/milvus/conf/server_config.yaml -l /var/lib/milvus/conf/log_config.conf' >> entrypoint.sh
RUN echo 'sleep 5' >> entrypoint.sh
RUN echo 'python3 -u run_algorithm.py "$@"' >> entrypoint.sh
RUN chmod u+x entrypoint.sh
ENTRYPOINT ["/home/app/entrypoint.sh"]
