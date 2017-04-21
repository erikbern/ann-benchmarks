#!/bin/sh

# success
../examples/test2 --version > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test8.out; then
	exit 0
else
	exit 1
fi

