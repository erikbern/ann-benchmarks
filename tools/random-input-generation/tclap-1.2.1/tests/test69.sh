#!/bin/sh

# Checks that parsing exceptions are properly
# propagated to the caller.
../examples/test18 --bob > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test69.out; then
	exit 0
else
	exit 1
fi

