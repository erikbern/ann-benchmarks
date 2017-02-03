#!/bin/sh

get_dataset() {
  url="$1"
  sha512sum="$2"
  filename="${url##*/}"
  unzipped="${filename%.gz}"
  if [ ! -f "$unzipped" ]; then
    if [ ! -f "$filename" ]; then
      wget "$url"
    fi
    sha512sum --check - <<END && gunzip -k "$filename"
$sha512sum $filename
END
  fi
}

get_dataset "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" \
    "0b15d987f9ca433ccf8079e8dc85b181fcad53747069ec9c1a879500155fa300c2984e266d1971ff961883df138855e208116837a391ff33aa43c86de7ecaa0b" && python convert_idx.py train-images-idx3-ubyte > mnist-data.txt
get_dataset "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" \
    "d68caa134766708d34187ad88c5f424d6e0a4ac5ddabf36b776084d32ab10530a63993174be77af40a714669ea06f05a369034e57c2bd5dcd0076abce6763e02" && python convert_idx.py t10k-images-idx3-ubyte > mnist-query.txt
