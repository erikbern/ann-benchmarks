FROM ann-benchmarks

RUN apt-get install -y libhdf5-openmpi-dev cython
RUN pip3 install nearpy bitarray redis sklearn
RUN python3 -c 'import nearpy'