#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

sh lib-annitu.sh --just-get || exit 1

ins_git_get https://github.itu.dk/maau/fcLSH &&
  gcc -O3 -marchive=native -c -o frontend.o ../lib-annitu/wrappers/frontend/frontend.c &&
  g++ -O3 -marchive=native -std=c++14 -o fr-fcLSH frontend.o ../lib-fcLSH.cpp -pthread &&
  g++ -O3 -marchive=native -std=c++14 -o fr-fcMIH frontend.o ../lib-fcMIH.cpp -pthread
