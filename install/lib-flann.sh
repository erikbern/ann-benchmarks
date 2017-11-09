#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require cmake &&
  ins_git_get https://github.com/mariusmuja/flann &&
  mkdir build &&
  cd build &&
  cmake .. &&
  make -j4 &&
  make install
