#!/bin/sh

# failure
../examples/test6 -n homer 6  > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test36.out; then
	exit 0
else
	exit 1
fi

