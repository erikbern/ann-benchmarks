# Adopted from https://github.com/microsoft/SPTAG/blob/master/Dockerfile

FROM ann-benchmarks

RUN git clone --recursive https://github.com/microsoft/SPTAG
RUN apt-get update && apt-get -y install wget build-essential cmake libboost-all-dev libtbb-dev software-properties-common swig

# Patch https://github.com/microsoft/SPTAG/issues/243
RUN cd SPTAG && \
        wget -qO- https://github.com/pabs3/SPTAG/commit/bd9c25d1409325ac45ebeb7f1e8fc87d03ec478c.patch | git apply && \
        cd ..

# SPTAG defaults to Python 2 if it's found on the system, so as a hack, we remove it. See https://github.com/microsoft/SPTAG/blob/master/Wrappers/CMakeLists.txt
RUN apt-get -y remove libpython2.7

# Compile
RUN cd SPTAG && mkdir build && cd build && cmake .. && make && cd ..

# so python can find the SPTAG module
ENV PYTHONPATH=/home/app/SPTAG/Release
RUN python3 -c 'import SPTAG'
