#!/bin/sh

# success
../examples/test1 -n mike -r > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test3.out; then
	exit 0
else
	exit 1
fi

