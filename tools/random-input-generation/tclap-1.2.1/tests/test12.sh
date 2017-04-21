#!/bin/sh

# failure
../examples/test2 -i 10 -s hello -f nine > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test12.out; then
	exit 0
else
	exit 1
fi

