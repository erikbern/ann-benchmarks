FROM ann-benchmarks

RUN apt-get update && apt-get install -y cmake pkg-config liblz4-dev
RUN git clone https://github.com/mariusmuja/flann
RUN mkdir flann/build
RUN cd flann/build && cmake ..
RUN cd flann/build && make -j4
RUN cd flann/build && make install
RUN pip3 install sklearn
RUN python3 -c 'import pyflann'
