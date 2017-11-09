#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

ins_deb_require libhdf5-openmpi-dev python-h5py cython &&
  ins_pip_get nearpy bitarray redis
