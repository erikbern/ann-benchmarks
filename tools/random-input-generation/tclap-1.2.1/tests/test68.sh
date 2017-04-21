#!/bin/sh

# this tests whether we can parse args from
# a vector of strings and that combined switch
# handling doesn't get fooled if the delimiter
# is in the string
# success  
../examples/test13 > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test68.out; then
	exit 0
else
	exit 1
fi

