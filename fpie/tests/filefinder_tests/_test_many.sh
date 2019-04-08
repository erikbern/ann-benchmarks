#!/usr/bin/env bash
set -eo pipefail

# Test case with many patterns
if diff <(../_filefinder.sh testdir filefinder_tests/.includefile_many | sort) <(cat filefinder_tests/expected_many.txt | sort) >/dev/null ; then
  exit 0
else
  exit 1
fi

