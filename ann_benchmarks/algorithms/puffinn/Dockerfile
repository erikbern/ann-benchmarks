FROM ann-benchmarks

RUN pip install pypandoc==1.4 pybind11
RUN git clone https://github.com/puffinn/puffinn
RUN cd puffinn && python setup.py install
RUN python -c 'import puffinn'
