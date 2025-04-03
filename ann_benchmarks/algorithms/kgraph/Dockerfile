FROM ann-benchmarks

RUN apt-get update && apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev libopenblas-dev
RUN git clone https://github.com/aaalgo/kgraph
RUN cd kgraph && pip install -e .
RUN python -c 'import kgraph'
