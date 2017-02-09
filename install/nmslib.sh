cd "$(dirname "$0")"
echo "Installing Python interface for the Non-Metric Space Library"

if [ ! -d "nmslib/.git" ]; then
  git clone https://github.com/searchivarius/nmslib.git
else (
  cd nmslib
  git reset --hard
  git clean -dfx
  git pull
) fi &&
  cd nmslib/similarity_search &&
  (apt-get install -y cmake libboost-all-dev libeigen3-dev libgsl0-dev || true) &&
  echo "CC: $CC, CXX: $CXX" &&
  export CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX &&
  export CMAKE_C_COMPILER_ENV_VAR=CC CMAKE_CXX_COMPILER_ENV_VAR=CXX &&
  cmake . -DWITH_EXTRAS=1 &&
  make -j4 &&
  cd ../python_bindings &&
  python setup.py build &&
  python setup.py install &&
  cd ../..
