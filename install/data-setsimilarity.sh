#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "aol-data.txt" ]; then
    if [ ! -f "ssjoin-data.tar.xz" ]; then
        wget http://ssjoin.dbresearch.uni-salzburg.at/downloads/projects/ssjoin/ssjoin-data.tar.xz
    fi

    tar xf ssjoin-data.tar.xz

    cp ssjoin-data/aol-data/aol-data-white-dedup-lenshuf-raw.txt aol-data.txt
    cat > aol-data.yaml <<END
    dataset:
      point_type: int
    test_size: 50
END

    cp ssjoin-data/flickr/flickr-dedup-lenshuf-raw.txt flickr-data.txt
    cat > flickr-data.yaml <<END
    dataset:
      point_type: int
    test_size: 50
END

    cp ssjoin-data/kosarek/kosarek-dedup-raw.txt kosarek-data.txt
    cat > kosarek-data.yaml <<END
    dataset:
      point_type: int
    test_size: 50
END
    rm ssjoin-data.tar.xz
    rm -rf ssjoin-data/
fi

