ARG QDRANT_VERSION=1.0.3
FROM qdrant/qdrant:v${QDRANT_VERSION}

RUN apt-get update
RUN apt-get install -y python3-pip build-essential python3-h5py
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /home/app

COPY requirements.txt run_algorithm.py ./

ARG QDRANT_VERSION
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install qdrant-client==${QDRANT_VERSION}

RUN echo '#!/bin/bash' >> entrypoint.sh
RUN echo 'cd /qdrant && ./qdrant &' >> entrypoint.sh
RUN echo 'sleep 5' >> entrypoint.sh
RUN echo 'python3 -u run_algorithm.py "$@"' >> entrypoint.sh
RUN chmod u+x entrypoint.sh
ENTRYPOINT ["/home/app/entrypoint.sh"]
