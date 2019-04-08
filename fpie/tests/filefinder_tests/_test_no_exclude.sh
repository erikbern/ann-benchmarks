#!/usr/bin/env bash
set -eo pipefail

# Test case with no exclude
if diff <(../_filefinder.sh testdir filefinder_tests/.includefile_no_exclude | sort) <(cat filefinder_tests/expected_no_exclude.txt | sort) >/dev/null ; then
  exit 0
else
  exit 1
fi

