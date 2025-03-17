FROM ann-benchmarks

RUN pip install pybind11
RUN git clone https://github.com/nmslib/hnsw.git;cd hnsw; git checkout denorm

RUN cd hnsw/python_bindings; python setup.py install

RUN python -c 'import hnswlib'

