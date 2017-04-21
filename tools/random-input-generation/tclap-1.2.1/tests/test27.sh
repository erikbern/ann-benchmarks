#!/bin/sh

# failure
../examples/test2 -i 2 -f 4.0.2 -s asdf asdf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test27.out; then
	exit 0
else
	exit 1
fi

