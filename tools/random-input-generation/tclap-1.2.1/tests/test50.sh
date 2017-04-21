#!/bin/sh

# failure
../examples/test8  -s one homer -B -Bh > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test50.out; then
	exit 0
else
	exit 1
fi

