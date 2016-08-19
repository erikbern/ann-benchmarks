cd "$(dirname "$0")"
git clone https://github.itu.dk/maau/ann-filters
sudo apt-get install -y swig
cd ann-filters 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../..
