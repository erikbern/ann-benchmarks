git clone https://github.com/aaalgo/kgraph
pushd kgraph
sudo make deps-ubuntu
make
make release
sudo cp python/pykgraph.so /usr/local/lib/python2.7/dist-packages
sudo cp kgraph-release/bin/libkgraph.so /usr/lib
cd ..
