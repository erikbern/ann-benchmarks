#!/bin/sh

# failure
../examples/test5 -a fdsa -b asdf -c fdas > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test22.out; then
	exit 0
else
	exit 1
fi

