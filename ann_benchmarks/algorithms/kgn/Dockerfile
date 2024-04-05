FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11 faiss-cpu

WORKDIR /home/app
RUN git clone https://github.com/Henry-yan/kgn.git
RUN pip3 install kgn/pykgn-1.0.0-cp310-cp310-linux_x86_64.whl