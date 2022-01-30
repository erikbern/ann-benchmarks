FROM ann-benchmarks

COPY requirements_py38.txt ./

RUN python3 -m pip install -r requirements_py38.txt

RUN pip3 install numpy --upgrade

RUN pip3 install vearch

RUN python3 -c 'import vearch'

