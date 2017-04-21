#!/bin/sh

# success test hex
../examples/test19 -i 0xA > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test71.out; then
	exit 0
else
	exit 1
fi

