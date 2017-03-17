#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev &&
  ins_git_get https://github.com/aaalgo/kgraph &&
  if [ -z "$CC" ]
  then
    CC=g++
  fi &&
  python setup.py build &&
  python setup.py install
