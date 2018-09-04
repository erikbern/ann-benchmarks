FROM ann-benchmarks

RUN apt-get update && apt-get install -y libopenblas-base libopenblas-dev libpython-dev python-numpy python-pip swig
RUN git clone https://github.com/facebookresearch/faiss lib-faiss
RUN cd lib-faiss && git checkout tags/v1.2.1 -b lib-faiss && cp example_makefiles/makefile.inc.Linux makefile.inc && make -j4 py BLASLDFLAGS=/usr/lib/x86_64-linux-gnu/libopenblas.so.0
ENV PYTHONPATH lib-faiss

# faiss doesn't work with python3 afaik
RUN python -c 'import faiss'
RUN pip install -r requirements.txt
RUN pip install sklearn enum34
ENTRYPOINT ["python", "run_algorithm.py"]
