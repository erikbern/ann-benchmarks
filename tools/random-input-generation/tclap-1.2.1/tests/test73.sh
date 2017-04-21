#!/bin/sh

# success  tests whether * in UnlabeledValueArg passes 
../examples/test2 -i 1 -s asdf fff*fff > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test73.out; then
	exit 0
else
	exit 1
fi

