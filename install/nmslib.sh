cd "$(dirname "$0")"
echo "Installing Python interface for the Non-Metric Space Library"
# Remove the previous version if existed
rm -rf nmslib 
# Note that we use version 1.5.2 here:
#git clone https://github.com/searchivarius/nmslib.git
wget https://github.com/searchivarius/nmslib/archive/v1.5.2.tar.gz
tar -zxvf v1.5.2.tar.gz
cd nmslib-1.5.2/similarity_search
apt-get install -y cmake libeigen3-dev libgsl0-dev
echo "CC: $CC, CXX: $CXX"
export CMAKE_C_COMPILER=$CC
export CMAKE_CXX_COMPILER=$CXX
export CMAKE_C_COMPILER_ENV_VAR=CC
export CMAKE_CXX_COMPILER_ENV_VAR=CXX
cmake .
make -j 4
cd ../python_vect_bindings
make CC=$CC
make install
cd ../..
