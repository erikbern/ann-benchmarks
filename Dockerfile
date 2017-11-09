FROM	ubuntu:16.04
RUN	apt-get update -y && apt-get install -y ruby git python-pip python-dev build-essential wget python-tk
RUN pip install matplotlib
ADD	. ann-benchmarks
WORKDIR	ann-benchmarks
RUN git config --global user.email "ann-benchmarks@ann-benchmarks.com"
RUN git config --global user.name "ANN Benchmarks"
RUN mkdir queries/
RUN	bash install.sh
