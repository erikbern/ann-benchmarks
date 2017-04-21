#!/bin/sh

# failure
../examples/test3 -f=9 -f=1.0.0 -s=asdf asdf asdf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test32.out; then
	exit 0
else
	exit 1
fi

