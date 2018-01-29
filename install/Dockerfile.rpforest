FROM ann-benchmarks

RUN git clone https://github.com/lyst/rpforest
RUN cd rpforest && python3 setup.py install
RUN python3 -c 'import rpforest'
