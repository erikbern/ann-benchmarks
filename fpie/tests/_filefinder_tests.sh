#!/usr/bin/env bash
set -eo pipefail

./filefinder_tests/_test_many.sh
./filefinder_tests/_test_no_include.sh
./filefinder_tests/_test_no_exclude.sh

