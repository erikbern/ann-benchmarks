#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require cmake libboost-all-dev libeigen3-dev libgsl0-dev &&
  ins_git_get https://github.com/searchivarius/nmslib.git &&
  cd similarity_search &&
  echo "CC: $CC, CXX: $CXX" &&
  export CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX &&
  export CMAKE_C_COMPILER_ENV_VAR=CC CMAKE_CXX_COMPILER_ENV_VAR=CXX &&
  cmake . -DWITH_EXTRAS=1 &&
  make -j4 &&
  cd ../python_bindings &&
  python setup.py build &&
  python setup.py install
