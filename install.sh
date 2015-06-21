apt-get update
apt-get install -y python-numpy python-scipy python-pip python-nose build-essential
pip install scikit-learn

# Install GCC 4.8
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update -qq
apt-get install -y libboost1.48-all-dev g++-4.8
export CXX="g++-4.8" CC="gcc-4.8"

cd install
for fn in annoy.sh panns.sh nearpy.sh sklearn.sh flann.sh kgraph.sh nmslib.sh
do
    source $fn
done
