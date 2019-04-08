FROM ubuntu:18.04 as build
WORKDIR /app
COPY . /app

FROM build as test
RUN cd /app/tests && ./_filepacker_tests.sh && ./_filefinder_tests.sh

FROM build
ENTRYPOINT [ "sh" ]
