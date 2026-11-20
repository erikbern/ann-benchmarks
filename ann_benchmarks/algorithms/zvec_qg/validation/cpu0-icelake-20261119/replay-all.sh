#!/usr/bin/env bash
set -euo pipefail
archive=$1
dest=$2
bench=$3
numa=$4
: "${QG_ISOLATION_MANIFEST:?run through qg_bench.py isolate}"
for ds in sift gist glove; do
    case "$ds" in
        sift) data=sift-128-euclidean;;
        gist) data=gist-960-euclidean;;
        glove) data=glove-100-angular;;
    esac
    for rep in 0 1 2; do
        "$numa" --physcpubind=0 --membind=0 "$bench/.venv/bin/python" "$archive/replay.py" "$ds" \
            --data "$bench/data/$data.hdf5" --repeat "$rep" --output "$dest/$ds-$rep"
    done
done
