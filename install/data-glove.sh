#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "glove.txt" ]; then
  ins_data_get \
      "https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz" \
      "054f2ea9f9b168ce1c205ca1b9b5b0a2" &&
    cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt && # strip first column
    ruby ../tools/split-dataset.rb glove.txt glove-data.txt glove-query.txt 10000 312322936
    rm -f glove.twitter.27B.100d.txt
    rm -f glove.txt
fi
