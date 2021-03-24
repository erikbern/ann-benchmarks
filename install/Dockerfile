FROM ubuntu:18.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

RUN apt-get update && apt-get install -y python3-numpy python3-scipy python3-pip build-essential git
RUN pip3 install -U pip

RUN addgroup --gid $GROUP_ID user; exit 0
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME; exit 0
RUN echo "$USER_NAME:$USER_NAME" | chpasswd && adduser $USER_NAME sudo
RUN echo '----->'
RUN echo 'root:Docker!' | chpasswd
# ugly
RUN chown -R $USER_NAME:$GROUP_ID /usr/local
WORKDIR /home/app
RUN chown -R $USER_NAME:$GROUP_ID /home/app
USER $USER_NAME

COPY requirements.txt run_algorithm.py ./
RUN pip3 install --user -rrequirements.txt


ENTRYPOINT ["python3", "run_algorithm.py"]
