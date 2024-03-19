FROM ann-benchmarks

RUN apt-get -y install curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://install.surrealdb.com | sh -s -- --nightly

RUN pip install requests

