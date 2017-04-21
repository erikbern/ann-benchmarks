#!/bin/sh

# success
../examples/test1 -r -n mike > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test1.out; then
	exit 0
else
	exit 1
fi

