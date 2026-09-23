FROM ann-benchmarks

WORKDIR /home/app
COPY . /home/app/

RUN python3 -m pip install --no-cache-dir numpy

CMD ["python3", "-u", "run_algorithm.py"]
