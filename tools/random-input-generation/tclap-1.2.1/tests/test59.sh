#!/bin/sh

# success
../examples/test9 -VVV -N --noise -r blah > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test59.out; then
	exit 0
else
	exit 1
fi

