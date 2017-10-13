#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "fashion-mnist-data.txt" ]; then
  ins_data_get \
      "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz" \
      "8d4fb7e6c68d591d4c3dfef9ec88bf0d" &&
    python convert_idx.py train-images-idx3-ubyte > fashion-mnist-data.txt &&
    rm -f train-images-idx3-ubyte
fi

if [ ! -f "fashion-mnist-query.txt" ]; then
  ins_data_get \
      "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz" \
      "bef4ecab320f06d8554ea6380940ec79" &&
    python convert_idx.py t10k-images-idx3-ubyte > fashion-mnist-query.txt &&
    rm -f t10k-images-idx3-ubyte
fi
