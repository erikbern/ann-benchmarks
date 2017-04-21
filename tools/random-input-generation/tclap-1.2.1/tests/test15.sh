#!/bin/sh

# failure
../examples/test3  --stringTest bbb homer marge bart -- -hv two > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test15.out; then
	exit 0
else
	exit 1
fi

