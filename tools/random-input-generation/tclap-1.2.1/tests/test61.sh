#!/bin/sh

# this tests a bug in handling of - chars in Unlabeled args
# success  
../examples/test2 -i 10 -s hello "-1 -1" > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test61.out; then
	exit 0
else
	exit 1
fi

