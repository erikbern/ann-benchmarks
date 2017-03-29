#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

sh lib-annitu.sh --just-get || exit 1

ins_git_get https://github.com/ipsarros/Dolphinn &&
  gcc -O3 -march=native -ffast-math -c -o frontend.o ../lib-annitu/wrappers/frontend/frontend.c &&
  g++ -O3 -march=native -ffast-math -std=c++14 -o fr-dolphinn frontend.o ../lib-dolphinn.cpp -pthread
