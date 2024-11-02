FROM ann-benchmarks

# Install dependencies for MariaDB build
RUN apt-get update
RUN apt-get install -yq --no-install-recommends libmariadb-dev devscripts equivs ccache ninja-build curl
RUN curl -LO https://github.com/MariaDB/server/raw/HEAD/debian/control
RUN mk-build-deps -r -i control -t 'apt-get -y -o Debug::pkgProblemResolver=yes --no-install-recommends'

# Install python MariaDB connector
RUN pip3 install mariadb

# Current development branch for vector support:
RUN curl -LO https://archive.mariadb.org//mariadb-11.6.0_vector/source/mariadb-11.6.0_vector.tar.gz
RUN tar -zxvf mariadb-11.6.0_vector.tar.gz
RUN mv mariadb-11.6.0_vector server

# Build flags from https://salsa.debian.org/mariadb-team/mariadb-server/-/blame/debian/latest/.gitlab-ci.yml?ref_type=heads#L43 , also skipped SPIDER as it failed to build in the branch
RUN cd server/ && cmake . -DWITH_SSL=system -DPLUGIN_COLUMNSTORE=NO -DPLUGIN_ROCKSDB=NO -DPLUGIN_S3=NO -DPLUGIN_MROONGA=NO -DPLUGIN_CONNECT=NO -DPLUGIN_MROONGA=NO -DPLUGIN_TOKUDB=NO -DPLUGIN_PERFSCHEMA=NO -DWITH_WSREP=OFF -DPLUGIN_SPIDER=NO
RUN cd server/ && make -j`nproc`
RUN cd server/ && make install

# Initialize database
WORKDIR /home/mysql
RUN /usr/local/mysql/scripts/mariadb-install-db
# Initialization done, skip running init when ann-benchmark is running in docker container
ENV DO_INIT_MARIADB 0

# Other environment variables for module.py
ENV MARIADB_ROOT_DIR /usr/local/mysql
ENV MARIADB_DB_WORKSPACE /home/mysql

WORKDIR /home/app
