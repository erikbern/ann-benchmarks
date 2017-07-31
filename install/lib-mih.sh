#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require cmake libhdf5-dev &&
  ins_git_get https://github.com/norouzi/mih &&
  mkdir build &&
  mkdir wrapper &&
  cp ../../protocol/c/frontend.c wrapper/ &&
  cp ../lib-mih-wrapper.cpp wrapper/ &&
  cd build &&
  cmake ../ &&
  make -j4


