FROM ann-benchmarks

RUN pip install pybind11 numpy setuptools
RUN apt-get update -y
RUN apt-get install -y g++ cmake libboost-dev libgoogle-perftools-dev
# RUN git clone https://github.com/twuebker/nsg.git

# RUN git fetch origin fix-distance
# RUN git checkout fix-distance
# RUN pip install .[knn]
RUN pip install pynsg[knn]

RUN python -c 'import pynsg'

