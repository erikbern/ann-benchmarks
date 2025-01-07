ARG QDRANT_VERSION=1.5.1
ARG QDRANT_CLIENT_VERSION=1.5.4
FROM qdrant/qdrant:v${QDRANT_VERSION}

ARG QDRANT_CLIENT_VERSION
WORKDIR /home/app
COPY requirements.txt .

RUN apt-get update \
 && apt-get install -y python3-pip build-essential python3-h5py \
 && python3 -m pip install --break-system-packages --upgrade pip setuptools wheel \
 && python3 -m pip install --break-system-packages -r requirements.txt \
 && python3 -m pip install --break-system-packages qdrant-client==${QDRANT_CLIENT_VERSION}

COPY run_algorithm.py .

RUN printf '#!/bin/bash\n\
cd /qdrant && ./qdrant &\n\
sleep 5\n\
python3 -u run_algorithm.py "$@"' > entrypoint.sh \
 && chmod u+x entrypoint.sh

ENTRYPOINT ["/home/app/entrypoint.sh"]
