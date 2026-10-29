FROM ann-benchmarks

RUN apt update
RUN apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev
RUN pip3 install wheel pybind11==2.13.6
RUN git clone https://github.com/intel/x86-simd-sort.git
RUN cd x86-simd-sort && git fetch --tags && git checkout v7.0
RUN pip3 install meson ninja
RUN cd x86-simd-sort && meson setup build -Dbuildtype=release && ninja -C build && meson install -C build
RUN git clone -b main https://github.com/NGT-labs/ngt.git
RUN mkdir -p ngt/build
RUN cd ngt/build && cmake -DNGTQG_NO_ROTATION=ON -DNGTQG_ZERO_GLOBAL=ON -DNGT_X86SIMDSORT=ON -DNGTQG_X86SIMDSORT=ON -DNGT_DISABLE_NORMALIZATION_ERROR_CHECK=ON ..
RUN cd ngt/build && make -j 4 && make install
RUN ldconfig
RUN cd ngt/python && python3 setup.py bdist_wheel
RUN pip3 install ngt/python/dist/ngt-*-linux_x86_64.whl
