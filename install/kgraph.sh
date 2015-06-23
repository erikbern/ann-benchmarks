git clone https://github.com/aaalgo/kgraph
apt-get install -y libboost-timer1.48-dev libboost-chrono1.48-dev libboost-program-options1.48-dev libboost-system1.48-dev
apt-get autoremove -y

cd kgraph
make CC=$CC libkgraph.so python
sudo cp python/pykgraph.so /usr/local/lib/python2.7/dist-packages
sudo cp libkgraph.so /usr/lib
cd ..
