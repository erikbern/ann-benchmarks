#!/bin/sh

# failure
../examples/test2 -i 2a -f 4.2 -s asdf asdf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test28.out; then
	exit 0
else
	exit 1
fi

