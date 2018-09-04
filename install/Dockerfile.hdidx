FROM ann-benchmarks

# needed to avoid some dependencies starting interaction on the command line
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python-opencv \
    python-numpy \
    python-pip \
    git
RUN pip install cython
RUN pip install -r requirements.txt

RUN git clone https://github.com/hdidx/hdidx.git
RUN cd hdidx && python setup.py install

RUN python -c 'import hdidx; a = hdidx.indexer.SHIndexer'
ENTRYPOINT ["python", "run_algorithm.py"]
