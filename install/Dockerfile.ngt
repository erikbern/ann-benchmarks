FROM ann-benchmarks

RUN apt-get update
RUN apt-get install -y git cmake g++ python3 python3-setuptools python3-pip
RUN pip3 install wheel pybind11
RUN git clone https://github.com/yahoojapan/ngt.git
RUN mkdir -p ngt/build
RUN cd ngt/build && cmake ..
RUN cd ngt/build && make && make install
RUN ldconfig
RUN cd ngt/python && python3 setup.py bdist_wheel
RUN pip3 install ngt/python/dist/ngt-*-linux_x86_64.whl

