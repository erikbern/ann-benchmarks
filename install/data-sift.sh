#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "sift.txt" ]; then
  ins_data_get \
      "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" \
      "b23d1b3b2ee8469d819b61ca900ef0ed" &&
    tar -xf sift.tar &&
    rm -f sift.tar &&
    python convert_texmex_fvec.py sift/sift_base.fvecs > sift-data.txt &&
    python convert_texmex_fvec.py sift/sift_query.fvecs > sift-query.txt &&
    head -n 50000 sift-data.txt > siffette.txt &&
    rm -rf sift
fi
