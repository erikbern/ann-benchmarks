FROM ann-benchmarks

WORKDIR /home/app

# Install temurin JDK
RUN apt-get update
RUN apt-get install -y wget apt-transport-https gnupg
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update
RUN apt-get install -y temurin-20-jdk temurin-17-jdk

# Install PyLucene & JCC
RUN wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.7.0-src.tar.gz
RUN tar -xzf pylucene-9.7.0-src.tar.gz
ENV JCC_JDK=/usr/lib/jvm/temurin-20-jdk-amd64
WORKDIR /home/app/pylucene-9.7.0/jcc
RUN python3 ./setup.py build
RUN python3 ./setup.py install

WORKDIR /home/app/pylucene-9.7.0
ENV PYTHON=python3
ENV JCC="$(PYTHON) -m jcc --shared"
ENV NUM_FILES=16
# Needed as current Lucene gradle version doesn't support java 20 on build
ENV JAVA_HOME=/usr/lib/jvm/temurin-17-jdk-amd64
RUN make lucene-java-9.7.0
RUN mkdir lucene-java-9.7.0/lucene/extensions/src/java/org/apache/pylucene/codecs
RUN wget -O ./lucene-java-9.7.0/lucene/extensions/src/java/org/apache/pylucene/codecs/PyLucene95Codec.java https://gist.githubusercontent.com/benwtrent/79d70d59716f0e25833c5ea84d956c12/raw/8f529f3437c2fb8318f0127ecd71c960e43e0a7f/PyLucene95Codec.java
RUN make
RUN make install
# Switch back to jdk20 for panama vectorization support
ENV JAVA_HOME=/usr/lib/jvm/temurin-20-jdk-amd64

# Reset the work dir so scripts can be ran
WORKDIR /home/app