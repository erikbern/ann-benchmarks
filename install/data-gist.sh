#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "gist-data.txt" ]; then
  ins_data_get \
      "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" \
      "31185e0f00854f74d27e8ad8d52628a9" &&
    tar -xf gist.tar &&
    rm -f gist.tar &&
    python convert_texmex_fvec.py gist/gist_base.fvecs > gist-data.txt &&
    python convert_texmex_fvec.py gist/gist_query.fvecs > gist-query.txt &&
    rm -rf gist
fi
