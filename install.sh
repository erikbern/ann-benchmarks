if [ -z $ALGORITHM ]
then
    ALGORITHM=$1
fi

cd install && docker build -t ann-benchmarks-$ALGORITHM -f Dockerfile.$ALGORITHM .
