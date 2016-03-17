apt-get update
apt-get install -y python-numpy python-scipy python-pip python-nose build-essential software-properties-common unzip

# Install GCC 4.8
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update -qq
apt-get install -y g++-4.8
apt-get install -y libboost1.55-all-dev || apt-get install -y libboost1.48-all-dev || apt-cache search libboost
apt-get autoremove -y
export CXX="g++-4.8" CC="gcc-4.8"

pip install scikit-learn

cd install
for fn in annoy.sh panns.sh nearpy.sh sklearn.sh flann.sh kgraph.sh nmslib.sh rpforest.sh falconn.sh
do
    source $fn
done
