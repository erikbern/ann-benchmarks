apt-get install -y python-numpy python-scipy python-sklearn
cd install
for fn in annoy.sh panns.sh nearpy.sh sklearn.sh flann.sh kgraph.sh nmslib.sh
do
    source $fn
done
