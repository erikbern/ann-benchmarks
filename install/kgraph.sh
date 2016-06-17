rm -rf kgraph
git clone https://github.com/aaalgo/kgraph
apt-get install -y libboost-timer-dev libboost-chrono-dev libboost-program-options-dev libboost-system-dev libboost-python-dev
apt-get autoremove -y

cd kgraph
if [ -z "$CC" ]
then
CC=g++
fi
python setup.py build
sudo python setup.py install
cd ..
