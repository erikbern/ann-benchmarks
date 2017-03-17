#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require libopenblas-dev &&
  ins_git_get https://github.com/facebookresearch/faiss.git &&
  cp example_makefiles/makefile.inc.Linux makefile.inc &&
  make -j4 py BLASLDFLAGS=/usr/lib/libopenblas.so.0
