#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "mnist-data.txt" ]; then
  ins_data_get \
      "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" \
      "f68b3c2dcbeaaa9fbdd348bbdeb94873" &&
    python convert_idx.py train-images-idx3-ubyte > mnist-data.txt &&
    rm -f train-images-idx3-ubyte
fi

if [ ! -f "mnist-query.txt" ]; then
  ins_data_get \
      "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" \
      "9fb629c4189551a2d022fa330f9573f3" &&
    python convert_idx.py t10k-images-idx3-ubyte > mnist-query.txt &&
    rm -f t10k-images-idx3-ubyte
fi
