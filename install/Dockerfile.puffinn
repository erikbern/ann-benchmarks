FROM ann-benchmarks

RUN pip3 install pypandoc
RUN git clone https://github.com/puffinn/puffinn
RUN cd puffinn && python3 setup.py install
RUN python3 -c 'import puffinn'
