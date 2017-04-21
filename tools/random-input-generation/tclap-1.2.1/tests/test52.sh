#!/bin/sh

# failure
../examples/test8 -i=9a -i=1 -s=asdf asdf asdf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test52.out; then
	exit 0
else
	exit 1
fi

