#!/bin/sh

# success
../examples/test4 -BA --Bs asdf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test19.out; then
	exit 0
else
	exit 1
fi

