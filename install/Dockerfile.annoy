FROM ann-benchmarks

RUN git clone https://github.com/spotify/annoy
RUN cd annoy && python3 setup.py install
RUN python3 -c 'import annoy'
