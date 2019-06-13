FROM ann-benchmarks

RUN git clone https://github.com/microsoft/SPTAG

RUN apt-get update && apt-get -y install wget build-essential libtbb-dev \
    # remove the following if you don't want to build the wrappers
    openjdk-8-jdk python3-pip swig

# cmake >= 3.12 is required
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.tar.gz" -q -O - \
        | tar -xz --strip-components=1 -C /usr/local

# specific version of boost
RUN wget "https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz" -q -O - \
        | tar -xz && \
        cd boost_1_67_0 && \
        ./bootstrap.sh && \
        ./b2 install && \
        # update ld cache so it finds boost in /usr/local/lib
        ldconfig && \
        cd .. && rm -rf boost_1_67_0

# build
RUN cd SPTAG && mkdir build && cd build && cmake .. && make && cd ..

# so python can find the SPTAG module
ENV PYTHONPATH=/home/app/SPTAG/Release
RUN python3 -c 'import SPTAG'