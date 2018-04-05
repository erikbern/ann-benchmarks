FROM ann-benchmarks

RUN apt-get update && apt-get install -y python-pip python-numpy python-scipy
RUN pip install panns

# panns doesn't work with python3 afaik
RUN python -c 'import panns'
RUN pip install -rrequirements.txt
RUN pip install enum34
ENTRYPOINT ["python", "run_algorithm.py"]
