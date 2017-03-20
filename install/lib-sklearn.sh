#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_pip_get scikit-learn
ins_pip_get scipy
