#!/bin/sh

# failure
../examples/test9 -VVV -N --noise -rr > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test60.out; then
	exit 0
else
	exit 1
fi

