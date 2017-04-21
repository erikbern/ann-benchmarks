#!/bin/sh

# success
../examples/test8 --help  > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test54.out; then
	exit 0
else
	exit 1
fi

