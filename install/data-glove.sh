#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

if [ ! -f "glove.txt" ]; then
  ins_data_get "https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz" \
      "f718840c4c8de6b474c8cd8953e2a193aa3b928033aa7530c61c07c6b30b2d64db721a7c0f9f8f04ce1b699dfcc4894103c3d958956670fabda8959ccb242e64" &&
    cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt && # strip first column
    ruby ../tools/split-dataset.rb glove.txt glove-data.txt glove-query.txt 10000 312322936
    rm -f glove.twitter.27B.100d.txt
    rm -f glove.txt
fi
