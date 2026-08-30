FROM ann-benchmarks

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libomp-dev \
    ninja-build

# Install Python build dependencies
RUN pip install --no-cache-dir \
    setuptools \
    pybind11 \
    wheel \
    "numpy>=1.26.0"

# Install deglib
RUN pip install --no-cache-dir "deglib>=0.2.5"

RUN python -c 'import deglib'
