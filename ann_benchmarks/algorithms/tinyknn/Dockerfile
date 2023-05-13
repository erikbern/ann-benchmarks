FROM ann-benchmarks

RUN git clone https://github.com/thomasahle/tinyknn.git
RUN cd tinyknn && python3 -m pip install .
RUN python3 -c 'import tinyknn'
