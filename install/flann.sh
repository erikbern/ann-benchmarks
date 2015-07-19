cd "$(dirname "$0")"
git clone https://github.com/mariusmuja/flann
sudo apt-get install -y cmake
cd flann
git checkout bd1410dff94ee001b04de4a26a8f3ecefa155cb0 # Last version that builds on Ubuntu 12.04
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..
