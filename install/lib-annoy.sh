#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require python-dev python-setuptools &&
  ins_git_get https://github.com/spotify/annoy &&
  python setup.py install

 gcc -O3 -c -o frontend.o ../lib-annitu/wrappers/frontend/frontend.c
 g++ -O3 -std=c++14 -o fr-annoy frontend.o ../lib-annoy.cpp -pthread
