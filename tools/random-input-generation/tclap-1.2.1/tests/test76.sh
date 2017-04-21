#!/bin/sh

# failure  validates that the correct error message
# is displayed for XOR'd args
../examples/test20 -ba > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test76.out; then
	exit 0
else
	exit 1
fi

