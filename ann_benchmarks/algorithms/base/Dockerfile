FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y python3-pip python-is-python3 build-essential git
RUN python --version | grep 'Python 3.12'

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-u", "run_algorithm.py"]
