FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3-pip python-is-python3 build-essential git
RUN python --version | grep 'Python 3.10'

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-u", "run_algorithm.py"]
