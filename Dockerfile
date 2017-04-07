FROM	ubuntu:16.04
RUN	apt-get update -y && apt-get install -y git python-pip python-dev build-essential wget python-tk
RUN pip install matplotlib
ADD	. ann-benchmarks
WORKDIR	ann-benchmarks
RUN git config user.email "ann-benchmarks@ann-benchmarks.com"
RUN git config user.name "ANN Benchmarks"
RUN mkdir queries/
RUN	bash install.sh
RUN python -m ann_benchmarks.main --dataset siffette --distance euclidean
RUN python createwebsite.py --dataset random.data.sketch --outputdir website/
RUN scp -r website maau@pkqs.net:www/
