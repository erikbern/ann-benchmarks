FROM ann-benchmarks

WORKDIR /home/app

# Install temurin JDK
RUN apt-get update
RUN apt-get install -y wget apt-transport-https gnupg
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update
RUN apt-get install -y temurin-17-jdk

# Install PyLucene & JCC
RUN wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.4.1-src.tar.gz
RUN tar -xzf pylucene-9.4.1-src.tar.gz
ENV JCC_JDK=/usr/lib/jvm/temurin-17-jdk-amd64
WORKDIR /home/app/pylucene-9.4.1/jcc
RUN python3 ./setup.py build
RUN python3 ./setup.py install

WORKDIR /home/app/pylucene-9.4.1
ENV PYTHON=python3
ENV JCC="$(PYTHON) -m jcc --shared"
ENV NUM_FILES=16
RUN make lucene-java-9.4.1
RUN mkdir lucene-java-9.4.1/lucene/extensions/src/java/org/apache/pylucene/codecs
RUN wget -O ./lucene-java-9.4.1/lucene/extensions/src/java/org/apache/pylucene/codecs/PyLucene94Codec.java https://gist.githubusercontent.com/benwtrent/f3a6c4a9ce9749e702285dc82f39a129/raw/4742cf91401103f86809655d5c708b833beae43f/PyLucene94Codec.java
RUN make
RUN make install

# Reset the work dir so scripts can be ran
WORKDIR /home/app