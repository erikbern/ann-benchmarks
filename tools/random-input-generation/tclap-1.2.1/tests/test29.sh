#!/bin/sh

# failure...  no hex here, but see test19.cpp for how to use hex 
../examples/test2 -i 0xA -f 4.2 -s asdf asdf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test29.out; then
	exit 0
else
	exit 1
fi

