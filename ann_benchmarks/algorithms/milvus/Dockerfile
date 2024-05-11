FROM ann-benchmarks

# Add Docker's official GPG key:
RUN apt-get update
RUN apt-get install ca-certificates curl
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update 
RUN apt-get install -y wget docker-ce-cli docker-compose-plugin

RUN pip3 install pymilvus==2.4.1
RUN python3 -c "from pymilvus import Collection"

# Install milvus standalone by docker-compose.yml
RUN wget https://github.com/milvus-io/milvus/releases/download/v2.4.1/milvus-standalone-docker-compose.yml -O docker-compose.yml
RUN sed -i 's|${DOCKER_VOLUME_DIRECTORY:-.}|/tmp|g' docker-compose.yml
