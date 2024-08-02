FROM ann-benchmarks

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10

#RUN apt-get update 
#RUN apt-get install -y g++ software-properties-common
RUN pip3 install pybind11 numpy

ARG CACHEBUST=1
RUN git clone -b annbench https://github.com/cmuparlay/ParlayANN.git 
RUN cd ParlayANN && git submodule update --init --recursive
RUN cd ParlayANN/python && bash compile.sh
#RUN cd ParlayANN/python && pip install -e .
#RUN python3 -c 'import parlaypy'
ENV PYTHONPATH=$PYTHONPATH:/home/app/ParlayANN/python
WORKDIR /home/app
