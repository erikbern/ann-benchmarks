#!/bin/sh

# success
../examples/test2 -i 10 -s homer marge bart lisa > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test5.out; then
	exit 0
else
	exit 1
fi

