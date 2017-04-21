#!/bin/sh

# success
../examples/test3 --stringTest=asdf - asdf zero one > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test55.out; then
	exit 0
else
	exit 1
fi

