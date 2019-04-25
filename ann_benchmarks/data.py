from __future__ import absolute_import
import numpy


def float_parse_entry(line):
    return [float(x) for x in line.strip().split()]


def float_unparse_entry(entry):
    return " ".join(map(str, entry))


def int_parse_entry(line):
    return frozenset([int(x) for x in line.strip().split()])


def int_unparse_entry(entry):
    return " ".join(map(str, map(int, entry)))


def bit_parse_entry(line):
    return [bool(int(x)) for x in list(line.strip()
                                       .replace(" ", "")
                                       .replace("\t", ""))]


def bit_unparse_entry(entry):
    return " ".join(map(lambda el: "1" if el else "0", entry))


type_info = {
    "float": {
        "type": numpy.float,
        "parse_entry": float_parse_entry,
        "unparse_entry": float_unparse_entry,
        "finish_entries": numpy.vstack
    },
    "bit": {
        "type": numpy.bool_,
        "parse_entry": bit_parse_entry,
        "unparse_entry": bit_unparse_entry
    },
    "int": {
        "type": numpy.object,
        "parse_entry": int_parse_entry,
        "unparse_entry": int_unparse_entry,
    },
}
