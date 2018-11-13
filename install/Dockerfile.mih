FROM ann-benchmarks
RUN apt-get update && apt-get install -y cmake libhdf5-dev
RUN git clone https://github.com/maumueller/mih
RUN cd mih && mkdir bin && cd bin && cmake ../ && make -j4
