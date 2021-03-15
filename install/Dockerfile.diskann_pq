FROM ann-benchmarks

RUN apt-get update
RUN apt-get install -y wget git cmake g++ libaio-dev libgoogle-perftools-dev clang-format-4.0 libboost-dev python3 python3-setuptools python3-pip
RUN pip3 install pybind11 numpy

RUN cd /tmp && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN cd /tmp && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN cd /tmp && rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN cd /tmp && sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
RUN apt-get update
RUN apt-get install -y intel-mkl-64bit-2020.0-088

RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 150

RUN echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf
RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf
RUN ldconfig
RUN echo "MKL_THREADING_LAYER=GNU" >> /etc/environment
RUN export LD_LIBRARY_PATH="$PATH:/opt/intel/compilers_and_libraries/linux/lib/intel64"
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/"

RUN git clone --single-branch --branch python_bindings_quantized https://github.com/microsoft/diskann
RUN mkdir -p diskann/build
RUN cd diskann/build && cmake -DCMAKE_BUILD_TYPE=Release ..
RUN cd diskann/build && make -j
RUN cd diskann/python && pip install -e .
RUN python3 -c 'import vamanapy'
