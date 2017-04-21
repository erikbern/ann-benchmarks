#!/bin/sh
if [ ! -f "random-euclidean-data.txt" ]; then  
    cd ../tools/random-input-generation
    mkdir -p build &&
    cd build &&
    cmake -DCMAKE_BUILD_TYPE=Release .. &&
    make -j4
    src/rand-euclidean -s 388527680 -o ../../../install/rand-euclidean-data.txt -q ../../../install/rand-euclidean-query.txt -n 1000000 -d 128 -p 500 -c 500 -g
    src/rand-euclidean -s 210728676 -o ../../../install/rand-angular-data.txt -q ../../../install/rand-angular-query.txt -n 1000000 -d 128 -p 500 -c 500 -g -N
fi
