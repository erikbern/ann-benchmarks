#!/bin/sh

# failure
../examples/test5 -d junk -c fdas > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test23.out; then
	exit 0
else
	exit 1
fi

