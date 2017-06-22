#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "nytimes.tfidf.256.data.txt" ]; then
    if [ ! -f "nytimes.tfidf.256.tar.bz2" ]; then
        wget http://www.itu.dk/people/maau/ann-benchmark-datasets/nytimes.tfidf.256.tar.bz2
    fi

    tar xf nytimes.tfidf.256.tar.bz2
fi
