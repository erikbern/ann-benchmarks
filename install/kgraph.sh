git clone https://github.com/aaalgo/kgraph
pushd kgraph
sudo make deps-ubuntu
make
<<<<<<< HEAD
sudo make install 
popd
=======
make release
cd kgraph-release/python
make
cd ..
sudo cp python/pykgraph.so /usr/local/lib/python2.7/dist-packages
sudo cp bin/libkgraph.so /usr/lib
cd ../..
>>>>>>> ac6faf37b3761ab3a6f733a3271150d7ce747e1c
