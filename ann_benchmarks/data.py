from typing import List, Union, Set, FrozenSet


def float_parse_entry(line: str) -> List[float]:
    return [float(x) for x in line.strip().split()]


def float_unparse_entry(entry: List[float]) -> str:
    return " ".join(map(str, entry))


def int_parse_entry(line: str) -> FrozenSet[int]:
    return frozenset([int(x) for x in line.strip().split()])


def int_unparse_entry(entry: Union[Set[int], FrozenSet[int]]) -> str:
    return " ".join(map(str, map(int, entry)))


def bit_parse_entry(line: str) -> List[bool]:
    return [bool(int(x)) for x in list(line.strip().replace(" ", "").replace("\t", ""))]


def bit_unparse_entry(entry: List[bool]) -> str:
    return " ".join(map(lambda el: "1" if el else "0", entry))