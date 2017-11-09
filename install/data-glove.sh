#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "glove.txt" ]; then
  ins_data_get \
      "https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz" \
      "054f2ea9f9b168ce1c205ca1b9b5b0a2" &&
    cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt && # strip first column
    rm -f glove.twitter.27B.100d.txt
fi
