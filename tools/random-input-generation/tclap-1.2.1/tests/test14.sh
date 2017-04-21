#!/bin/sh

# success
../examples/test3  --stringTest=aaa homer marge bart -- one two > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test14.out; then
	exit 0
else
	exit 1
fi

