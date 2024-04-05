FROM ann-benchmarks

ENV APP_USERNAME=
ENV APP_PASSWORD=
ENV PROXIMUS_HOST='172.28.86.158'
ENV PROXIMUS_PORT=5000
ENV PROXIMUS_ADVERTISED_LISTENER=
ENV PROXIMUS_NAMESPACE='test'
ENV PROXIMUS_SET='ANN-data'
ENV VERIFY_TLS=
ENV APP_INDEXER_PARALLELISM=1


ARG AEROSPIKE_CLIENT_VERSION=0.3.1

RUN pip install pythonping
RUN python3 -m pip install --break-system-packages aerospike-proximus==${AEROSPIKE_CLIENT_VERSION}  --extra-index-url https://aerospike.jfrog.io/artifactory/api/pypi/aerospike-pypi-dev/simple

