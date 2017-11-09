#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh
if [ ! -f "random-euclidean-data.txt" ]; then
    ins_git_get https://github.com/maumueller/random-inputs &&
    mkdir -p build &&
    cd build &&
    cmake -DCMAKE_BUILD_TYPE=Release .. &&
    make -j4
    src/rand-euclidean -s 388527680 -o ../../rand-euclidean-data.txt -q ../../rand-euclidean-query.txt -n 1000000 -d 128 -p 500 -c 500 -g
    src/rand-euclidean -s 210728676 -o ../../rand-angular-data.txt -q ../../rand-angular-query.txt -n 1000000 -d 128 -p 500 -c 500 -g -N
fi
