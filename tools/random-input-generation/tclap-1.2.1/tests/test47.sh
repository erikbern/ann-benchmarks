#!/bin/sh

# success
../examples/test8 -s=bill -i=9 -i=8 -B homer marge bart > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test47.out; then
	exit 0
else
	exit 1
fi

