#!/bin/sh

# failure
../examples/test3  --stringTest one homer -B -Bh > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test16.out; then
	exit 0
else
	exit 1
fi

