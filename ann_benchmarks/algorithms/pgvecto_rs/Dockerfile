FROM tensorchord/pgvecto-rs:pg16-v0.3.0-alpha.1

# https://github.com/tensorchord/pgvecto.rs

RUN apt-get update \
 && apt-get install -y python3-pip

WORKDIR /home/app
COPY requirements.txt  .

RUN python3 -m pip install --break-system-packages -r requirements.txt
RUN python3 -m pip install --break-system-packages psycopg[binary]

COPY run_algorithm.py .

ENV POSTGRES_PASSWORD=password
ENV POSTGRES_USER=postgres

RUN printf '#!/bin/bash\n\
runuser -u postgres -- initdb \n\
runuser -u postgres -- postgres -c shared_preload_libraries=vectors.so &\n\
sleep 5\n\
python3 -u run_algorithm.py "$@"' > entrypoint.sh \
 && chmod u+x entrypoint.sh

ENTRYPOINT ["/home/app/entrypoint.sh"]
