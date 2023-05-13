FROM ann-benchmarks

RUN apt-get update && apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev
RUN git clone https://github.com/aaalgo/kgraph
RUN cd kgraph && python3 setup.py build && python3 setup.py install
RUN python3 -c 'import pykgraph'
