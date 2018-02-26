FROM ann-benchmarks

RUN apt-get update && apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev python-numpy python-pip
RUN git clone https://github.com/aaalgo/kgraph
RUN cd kgraph && python setup.py build && python setup.py install

# kgraph doesn't work with python3 afaik
RUN python -c 'import pykgraph'
RUN pip install -rrequirements.txt
RUN pip install enum34
ENTRYPOINT ["python", "run_algorithm.py"]
