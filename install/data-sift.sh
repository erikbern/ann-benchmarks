#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "sift.txt" ]; then
  ins_data_get \
      "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" \
      "1c0c5dcb9d8cad4754ddec94985c0a4c4230f755bce4cd2b93efa9a84d5b29477b496e707ba5ca37804f7c7616d5ce7e5be9a6241ffe654c4fd7be295f0e6224" &&
    tar -xf sift.tar &&
    rm -f sift.tar &&
    python convert_texmex_fvec.py sift/sift_base.fvecs > sift.txt &&
    head -n 50000 sift.txt > siffette.txt && 
    rm -rf sift
fi
