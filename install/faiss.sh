#!/bin/sh

if [ ! -d "faiss" ]; then
	git clone https://github.com/facebookresearch/faiss.git
else (
	cd faiss
	git clean -dfx
	git reset --hard
	git pull
) fi
cd faiss
cp example_makefiles/makefile.inc.Linux makefile.inc
sed --in-place "s@ -Dnullptr=NULL @ @" makefile.inc
make -j4 py BLASLDFLAGS=/usr/lib/libopenblas.so.0
