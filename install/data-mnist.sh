#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "mnist-data.txt" ]; then
  ins_data_get "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" \
      "0b15d987f9ca433ccf8079e8dc85b181fcad53747069ec9c1a879500155fa300c2984e266d1971ff961883df138855e208116837a391ff33aa43c86de7ecaa0b" &&
    python convert_idx.py train-images-idx3-ubyte > mnist-data.txt &&
    rm -f train-images-idx3-ubyte
fi

if [ ! -f "mnist-query.txt" ]; then
  ins_data_get "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" \
      "d68caa134766708d34187ad88c5f424d6e0a4ac5ddabf36b776084d32ab10530a63993174be77af40a714669ea06f05a369034e57c2bd5dcd0076abce6763e02" &&
    python convert_idx.py t10k-images-idx3-ubyte > mnist-query.txt &&
    rm -f t10k-images-idx3-ubyte
fi
