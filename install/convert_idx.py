#!/usr/bin/env python

# This script is free software; its author gives unlimited permission to copy,
# distribute and modify it.

import io
import sys
import argparse
import operator
from struct import unpack

type_code_info = {
  0x08: (1, "!B"),
  0x09: (1, "!b"),
  0x0B: (2, "!H"),
  0x0C: (4, "!I"),
  0x0D: (4, "!f"),
  0x0E: (8, "!d")
}

def entry_as_bv_string(e):
    " ".join(map(lambda e: bin(e)[2:].ljust(8, "0"), entry))

def main():
    parser = argparse.ArgumentParser(description="""\
Produce a plain text representation of an IDX data file, with one line per
entry. Entries will be flattened to a single dimension.""")
    parser.add_argument(
        "inpath",
        metavar="INFILE",
        help="an IDX data file")
    parser.add_argument(
        "-b", "--bit-vector",
        help="represent entries as bit vectors",
        action="store_true")

    args = parser.parse_args()
    inpath = args.inpath

    with io.open(inpath, "rb") as infile:
        magic, type_code, dim_count = unpack("!hBB", infile.read(4))
        assert magic == 0, """\
%s: magic number test failed""" % inpath
        assert type_code in type_code_info, """\
%s: unrecognised type code %d""" % (inpath, type_code)

        if args.bit_vector:
            assert type_code in [0x08, 0x09, 0x0B, 0x0C], """\
%s: the --bit-vector option is not supported for type code \
%d""" % (inpath, type_code)

        dimensions = []
        for i in xrange(dim_count):
            dimensions.append(unpack("!I", infile.read(4))[0])
        entry_count = dimensions[0]
        entry_size = reduce(operator.mul, dimensions[1:])

        bytes, format_string = type_code_info[type_code]
        for i in xrange(entry_count):
            entry = []
            for j in xrange(entry_size):
                part = unpack(format_string, infile.read(bytes))[0]
                entry.append(part)
            if args.bit_vector:
                print "".join(map(
                    lambda e: bin(e)[2:].ljust(bytes * 8, "0"), entry))
            else:
                print " ".join(map(str, entry))

if __name__ == "__main__":
    main()
