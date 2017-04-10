#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "gist-data.txt" ]; then
  ins_data_get \
      "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" \
      "3bed5dc38ef4b10805ce91c13c121fbd261440f5c4685a489a04748b2c0e4eddb7fd8645b4e47ce54b142062c134bbaea1e3fdb3a4ef178de3b08e0d3a910e78" &&
    tar -xf gist.tar &&
    rm -f gist.tar &&
    python convert_texmex_fvec.py gist/gist_base.fvecs > gist-data.txt &&
    python convert_texmex_fvec.py gist/gist_query.fvecs > gist-query.txt &&
    rm -rf gist
fi
