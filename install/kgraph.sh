rm -rf kgraph
git clone https://github.com/aaalgo/kgraph
apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev
apt-get autoremove -y

cd kgraph
if [ -z "$CC" ]
then
CC=g++
fi
make CC=$CC libkgraph.so python
sudo cp python/pykgraph.so /usr/local/lib/python2.7/dist-packages
sudo cp libkgraph.so /usr/lib
cd ..
