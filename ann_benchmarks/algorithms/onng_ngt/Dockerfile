FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11==2.13.6
RUN git clone -b main https://github.com/NGT-labs/ngt.git
RUN mkdir -p ngt/build
RUN cd ngt/build && cmake -DNGTQG_NO_ROTATION=ON -DNGTQG_ZERO_GLOBAL=ON -DNGT_DISABLE_NORMALIZATION_ERROR_CHECK=ON ..
RUN cd ngt/build && make && make install
RUN ldconfig
RUN cd ngt/python && python3 setup.py bdist_wheel
RUN pip3 install ngt/python/dist/ngt-*-linux_x86_64.whl
