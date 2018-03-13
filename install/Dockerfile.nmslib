FROM ann-benchmarks

RUN apt-get update && apt-get install -y cmake libboost-all-dev libeigen3-dev libgsl0-dev
RUN git clone https://github.com/searchivarius/nmslib.git
RUN cd nmslib/similarity_search && cmake . -DWITH_EXTRAS=1
RUN cd nmslib/similarity_search && make -j4
RUN apt-get install -y python-setuptools python-pip python-numpy
RUN pip install pybind11
RUN cd nmslib/python_bindings && python setup.py build
RUN cd nmslib/python_bindings && python setup.py install

# nmslib doesn't work with python3 afaik
RUN python -c 'import nmslib'
RUN pip install -rrequirements.txt
RUN pip install enum34
ENTRYPOINT ["python", "run_algorithm.py"]
