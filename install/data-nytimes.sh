#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "nytimes.tfidf.256.data.txt" ]; then
    if [ ! -f "nytimes.tifidf.256.data.tar.bz2" ]; then
        wget http://www.itu.dk/people/maau/ann-benchmark-datasets/nytimes.tifidf.256.data.tar.bz2
    fi

    tar xf nytimes.tifidf.256.data.tar.bz2
END


