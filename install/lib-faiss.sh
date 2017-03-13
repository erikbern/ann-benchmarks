#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_git_get https://github.com/facebookresearch/faiss.git &&
  make -j4 py BLASLDFLAGS=/usr/lib/libopenblas.so.0
