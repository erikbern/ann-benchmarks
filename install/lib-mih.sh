#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require cmake libhdf5-dev &&
  ins_git_get https://github.com/norouzi/mih &&
  mkdir build &&
  cd build &&
  cmake ../ &&
  make -j4
