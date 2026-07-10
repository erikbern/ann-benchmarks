FROM ann-benchmarks

RUN apt-get update
RUN apt-get install -y make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev python3.10-venv
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libmkl-full-dev

RUN pip install diskannpy==0.7.0

RUN python -c 'import diskannpy'
