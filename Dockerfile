FROM	ubuntu:16.04
RUN	apt-get update -y && apt-get install -y git python-pip python-dev build-essential wget
ADD	. ann-benchmarks
WORKDIR	ann-benchmarks
RUN	bash install.sh
RUN python -m ann_benchmarks.main --dataset siffette --distance euclidean
RUN python createwebsite.py --dataset random.data.sketch --outputdir website/
RUN scp -r website maau@pkqs.net:www/
