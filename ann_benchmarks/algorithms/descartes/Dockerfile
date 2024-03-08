FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11==2.5.0

WORKDIR /home/app
RUN git clone https://github.com/xiaoming-01ai/descartes.git
RUN pip3 install descartes/descartes-*-linux_x86_64.whl
