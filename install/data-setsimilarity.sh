#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -d "ssjoin-data" ]; then
    wget http://ssjoin.dbresearch.uni-salzburg.at/downloads/projects/ssjoin/ssjoin-data.tar.xz
    tar xz ssjoin-data.tar.xz
    cp aol-data/aol-data-white-dedup-lenshuf-raw.txt aol-data.txt
    cat > aol-data.yaml <<END
dataset:
  point_type: set
END
fi
