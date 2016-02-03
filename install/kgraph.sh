git clone https://github.com/aaalgo/kgraph
apt-get install -y libboost-timer1.48-dev libboost-chrono1.48-dev libboost-program-options1.48-dev libboost-system1.48-dev libopenblas-dev
apt-get autoremove -y

ls /usr/lib/openblas-base
if [ ! -f /usr/lib/openblas-base ]; then
   sudo mkdir /usr/lib/openblas-base
   sudo ln -s /usr/lib/libblas.a /usr/lib/openblas-base
fi

cd kgraph
make CC=$CC libkgraph.a python
sudo cp python/pykgraph.so /usr/local/lib/python2.7/dist-packages
cd ..
