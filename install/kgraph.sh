git clone https://github.com/aaalgo/kgraph
sudo apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev
cd kgraph
make
make release
cd kgraph-release/python
make
cd ..
sudo cp python/pykgraph.so /usr/local/lib/python2.7/dist-packages
sudo cp bin/libkgraph.so /usr/lib
cd ../..
