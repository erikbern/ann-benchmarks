FROM ubuntu:14.04
RUN apt-get update
RUN apt-get install -y git python-pip python-dev build-essential wget
RUN git clone https://github.com/erikbern/ann-benchmarks
RUN cd ann-benchmarks && bash install.sh
