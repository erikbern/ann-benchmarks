#!/bin/sh

# success
../examples/test2 -i 10 -s hello goodbye -hABC > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test7.out; then
	exit 0
else
	exit 1
fi

