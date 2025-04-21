FROM ann-benchmarks AS builder

RUN apt-get update
RUN apt-get install -y openjdk-11-jre-headless

RUN apt-get install -y curl sudo
ENV CASSANDRA_VERSION 4.1.0
ENV CASSANDRA_HOME /opt/cassandra

RUN curl -L https://downloads.apache.org/cassandra/$CASSANDRA_VERSION/apache-cassandra-$CASSANDRA_VERSION-bin.tar.gz

WORKDIR /home/app