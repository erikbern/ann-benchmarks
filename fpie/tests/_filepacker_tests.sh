#!/usr/bin/env bash
set -eo pipefail

TMPDIR=`mktemp -d`

cat _filepacker_test_input.txt | ../_filepacker.sh testdir > $TMPDIR/tst

if diff <(cd $TMPDIR && tar -xf tst && rm tst && find -L | sort) <(cat _filepacker_test_expected.txt | sort) >/dev/null ; then
  exit 0
else
  exit 1
fi
