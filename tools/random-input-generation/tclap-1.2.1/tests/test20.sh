#!/bin/sh

# success
../examples/test5 -a asdf -c fdas --eee blah -i sss -i fdsf > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test20.out; then
	exit 0
else
	exit 1
fi

