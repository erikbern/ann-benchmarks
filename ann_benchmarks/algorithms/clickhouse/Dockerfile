FROM ann-benchmarks

# Install prerequisite packages
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg

# Download the ClickHouse GPG key and store it in the keyring
RUN curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg

# Add the ClickHouse repository to apt sources
RUN echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=`dpkg --print-architecture`]  https://packages.clickhouse.com/deb stable main" | tee /etc/apt/sources.list.d/clickhouse.list

ARG DEBIAN_FRONTEND=noninteractive

# Update apt package lists and install
RUN apt-get update && apt-get install -y clickhouse-server clickhouse-client
RUN apt-get update && apt-get install -y sudo

RUN pip3 install clickhouse-connect

RUN clickhouse-client --version
