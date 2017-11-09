#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "nytimes-hamming-data.txt" ]; then
    if [ ! -f "hamming-space.tar.bz2" ]; then
        wget http://sss.projects.itu.dk/ann-benchmarks/datasets/hamming-space.tar.bz2
    fi

    tar xf hamming-space.tar.bz2

    cp hamming-space/nytimes.hamming.128.data nytimes-hamming-data.txt
    cat > nytimes-hamming-data.yaml <<END
    dataset:
      point_type: bit
    test_size: 10000
END

    cp hamming-space/pubmed.hamming.256.data pubmed-hamming-data.txt
        cat > pubmed-hamming-data.yaml <<END
        dataset:
          point_type: bit
        test_size: 10000
END

    cp hamming-space/sift.hamming.256.data sift-hamming-data.txt
        cat > sift-hamming-data.yaml <<END
        dataset:
          point_type: bit
        test_size: 10000
END

    rm hamming-space.tar.bz2
    rm -rf hamming-space/
fi

