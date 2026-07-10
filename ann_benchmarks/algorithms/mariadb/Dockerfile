FROM ann-benchmarks

RUN apt-get update

# Install Python mariadb module
RUN apt-get install -y libmariadb-dev
RUN pip3 install mariadb

# Install server
RUN apt-get install -y curl sudo
RUN curl -LsS https://r.mariadb.com/downloads/mariadb_repo_setup | sudo bash -s -- --mariadb-server-version="mariadb-11.8"
RUN apt-get install -y mariadb-server

WORKDIR /home/app
