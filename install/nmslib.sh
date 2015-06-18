echo "Installing Python interface for the Non-Metric Space Library"
# Remove the previous version if existed
rm -rf NonMetricSpaceLib 
# Note that we use the develop branch here:
git clone https://github.com/searchivarius/NonMetricSpaceLib.git
cd NonMetricSpaceLib/similarity_search
git checkout ann-benchmark
apt-get install -y cmake libeigen3-dev libgsl0-dev libboost-all-dev
echo "CC: $CC, CXX: $CXX"
cmake .
make -j 4
cd ../python_binding
make
make install
cd ../..
