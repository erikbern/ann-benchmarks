FROM ann-benchmarks

RUN pip3 install numba scikit-learn
RUN git clone https://github.com/lmcinnes/pynndescent
RUN cd pynndescent && python3 setup.py install
RUN python3 -c 'import pynndescent'
