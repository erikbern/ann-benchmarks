if [ -z $LIBRARY ]
then
    LIBRARY=$1
fi

docker build -t ann-benchmarks -f install/Dockerfile .
docker build -t ann-benchmarks-$LIBRARY -f install/Dockerfile.$LIBRARY . 
