#!/bin/sh

# this tests whether all required args are listed as
# missing when no arguments are specified
# failure  
../examples/test12 -v "1 2 3" -v "4 5 6" -v "7 8 9" -v "-1 0.2 0.4" \
    > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test65.out; then
	exit 0
else
	exit 1
fi

