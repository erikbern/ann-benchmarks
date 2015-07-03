FROM	ubuntu:14.04
RUN	apt-get update -y && apt-get install -y git python-pip python-dev build-essential wget
ADD	. ann-benchmarks
WORKDIR	ann-benchmarks
RUN	bash install.sh
RUN	nosetests
