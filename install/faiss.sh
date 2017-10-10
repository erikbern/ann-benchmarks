cd "$(dirname "$0")"

apt-get install -y python-dev python-setuptools libopenblas-dev python-numpy liblapack3

git clone https://github.com/facebookresearch/faiss
cd faiss

if [ -e /usr/lib/libopenblas.so.0 ]; then
    if [ -e /usr/lib/lapack/liblapack.so.3.0 ]; then
        # handle ubuntu v14
        export BLASLDFLAGS='/usr/lib/libopenblas.so.0 /usr/lib/lapack/liblapack.so.3.0'
    else
        export BLASLDFLAGS=/usr/lib/libopenblas.so.0
    fi
fi
cp example_makefiles/makefile.inc.Linux makefile.inc

make
make py

cd ..
