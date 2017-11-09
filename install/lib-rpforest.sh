#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require python-dev python-setuptools &&
  ins_git_get https://github.com/lyst/rpforest &&
  python setup.py install
