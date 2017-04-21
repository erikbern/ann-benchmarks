#!/bin/sh

# success
../examples/test2 -i 1 - -s fdsa one two > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test56.out; then
	exit 0
else
	exit 1
fi

