#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require python-sklearn python-scipy
