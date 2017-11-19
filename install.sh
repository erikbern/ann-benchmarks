if [ -z $ALGORITHM ]
then
    ALGORITHM=$1
fi

docker build -t ann-benchmarks -f install/Dockerfile .
docker build -t ann-benchmarks-$ALGORITHM -f install/Dockerfile.$ALGORITHM . 
