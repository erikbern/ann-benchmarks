FROM centos:7

RUN yum -y install epel-release && \
    yum -y install centos-release-scl && \
    yum -y --setopt=skip_missing_names_on_install=False install gcc make git python3-devel && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    yum-config-manager --add-repo https://copr.fedorainfracloud.org/coprs/g/vespa/vespa/repo/epel-7/group_vespa-vespa-epel-7.repo && \
    yum -y --setopt=skip_missing_names_on_install=False --enablerepo=epel-testing install vespa-ann-benchmark

WORKDIR /home/app

COPY requirements.txt run_algorithm.py ./

RUN python3 -m pip install -r requirements.txt && \
    python3 -m pip install /opt/vespa/libexec/vespa_ann_benchmark

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
