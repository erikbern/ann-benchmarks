#!/usr/bin/env bash
set -eo pipefail

# Pack the filelist in a tar file.
RELATIVEPATH=`realpath $1`
tar -C ${RELATIVEPATH} -czf - --no-recursion --files-from=- | cat
