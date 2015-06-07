sudo apt-get install -y python-numpy python-scipy
cd install
for fn in annoy.sh panns.sh nearpy.sh sklearn.sh flann.sh kgraph.sh glove.sh sift.sh
do
    source $fn
done
