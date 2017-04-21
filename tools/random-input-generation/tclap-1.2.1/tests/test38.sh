#!/bin/sh

# failure
../examples/test7 -n mike 2 1  > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test38.out; then
	exit 0
else
	exit 1
fi

