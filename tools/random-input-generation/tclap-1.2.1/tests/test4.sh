#!/bin/sh

# failure
../examples/test1 > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test4.out; then
	exit 0
else
	exit 1
fi

