#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "nytimes.tfidf.256.data.txt" ]; then
    if [ ! -f "nytimes.tifidf.256.data.tar.bz2" ]; then
        wget http://sss.projects.itu.dk/ann-benchmarks/datasets/nytimes.tfidf.256.tar.bz2
    fi

    tar xf nytimes.tfidf.256.tar.bz2 && rm nytimes.tfidf.256.tar.bz2
fi


