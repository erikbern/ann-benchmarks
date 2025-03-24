FROM almalinux:8

RUN dnf -y install epel-release && \
    dnf -y install dnf-plugins-core && \
    dnf -y install \
        gcc \
        make \
        git \
        python3.11-devel \
        python3.11-pip \
        python3.11-setuptools && \
    dnf config-manager --add-repo https://raw.githubusercontent.com/vespa-engine/vespa/master/dist/vespa-engine.repo && \
    dnf -y install --enablerepo=powertools vespa-ann-benchmark

WORKDIR /home/app

COPY requirements.txt run_algorithm.py ./

RUN python3.11 -m pip install -r requirements.txt && \
    python3.11 -m pip install /opt/vespa/libexec/vespa_ann_benchmark

ENTRYPOINT ["python3.11", "-u", "run_algorithm.py"]
