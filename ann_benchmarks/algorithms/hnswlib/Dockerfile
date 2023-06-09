FROM ann-benchmarks

RUN apt-get install -y python-setuptools python-pip
RUN pip3 install pybind11 numpy setuptools
RUN git clone https://github.com/nmslib/hnsw.git;cd hnsw; git checkout denorm

RUN cd hnsw/python_bindings; python3 setup.py install

RUN python3 -c 'import hnswlib'

