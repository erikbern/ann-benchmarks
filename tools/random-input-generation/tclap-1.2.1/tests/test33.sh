#!/bin/sh

# success
../examples/test5 -a asdf -c fdas --eee blah --ddd -j o --jjj t > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test33.out; then
	exit 0
else
	exit 1
fi

