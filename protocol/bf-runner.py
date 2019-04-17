from __future__ import print_function

import sys
from enum import Enum
from shlex import split

from ann_benchmarks.data import type_info
from ann_benchmarks.distance import metrics
from ann_benchmarks.algorithms.bruteforce import BruteForce, BruteForceBLAS


class QueryMode(Enum):
    NORMAL = 0,
    PREPARED = 1,
    BATCH = 2


__true_print = print


def print(*args, **kwargs):
    __true_print(*args, **kwargs)
    sys.stdout.flush()


def next_line():
    for line in iter(sys.stdin.readline, ''):
        yield split(line.strip())


if __name__ == '__main__':
    point_type = None
    distance = None
    query_mode = QueryMode.NORMAL
    fast = False
    query_parameters = False
    # Configuration mode
    for line in next_line():
        if not line:
            break
        elif len(line) == 2:
            var, val = line[0], line[1]
            if var == "point-type":
                if val in type_info:
                    point_type = type_info[val]
                    print("epbprtv0 ok")
                else:
                    print("epbprtv0 fail")
            elif var == "distance":
                if val in metrics:
                    distance = val
                    print("epbprtv0 ok")
                else:
                    print("epbprtv0 fail")
            elif var == "fast":
                fast = (val == "1")
                print("epbprtv0 ok")
            else:
                print("epbprtv0 fail")
        elif len(line) == 3 and line[0] == "frontend":
            var, val = line[1], line[2]
            if var == "prepared-queries":
                query_mode = \
                    QueryMode.PREPARED if val == "1" else QueryMode.NORMAL
                print("epbprtv0 ok")
            elif var == "batch-queries":
                query_mode = \
                    QueryMode.BATCH if val == "1" else QueryMode.NORMAL
                print("epbprtv0 ok")
            elif var == "query-parameters":
                query_parameters = (val == "1")
                print("epbprtv0 ok")
            else:
                print("epbprtv0 fail")
        else:
            print("epbprtv0 fail")
    if point_type and distance:
        print("epbprtv0 ok")
    else:
        print("epbprtv0 fail")
        sys.exit(1)

    obj = None
    if not fast:
        obj = BruteForce(distance)
    else:
        obj = BruteForceBLAS(distance)

    parser = point_type["parse_entry"]
    # Training mode
    points = []
    for line in next_line():
        if not line:
            break
        elif len(line) == 1:
            point = line[0]
            try:
                parsed = parser(point)
                print("epbprtv0 ok %d" % len(points))
                points.append(parsed)
            except ValueError:
                print("epbprtv0 fail")
        else:
            print("epbprtv0 fail len %d" % len(line))
    if "finish_entries" in point_type:
        points = point_type["finish_entries"](points)
    obj.fit(points)
    print("epbprtv0 ok %d" % len(points))

    def _query_parameters(line):
        if hasattr(obj, "set_query_arguments"):
            try:
                obj.set_query_arguments(*line[1:-1])
                print("epbprtv0 ok")
            except TypeError:
                print("epbprtv0 fail")
        else:
            print("epbprtv0 fail")

    if query_mode == QueryMode.NORMAL:
        # Query mode
        for line in next_line():
            if not line:
                break
            elif query_parameters and line[0] == "query-params" \
                    and line[-1] == "set":
                _query_parameters(line)
            elif len(line) == 2:
                try:
                    query_point, k = line[0], int(line[1])
                    parsed = parser(query_point)
                    results = obj.query(parsed, k)
                    if results:
                        print("epbprtv0 ok %d" % len(results))
                        for index in results:
                            print("epbprtv0 %d" % index)
                    else:
                        print("epbprtv0 fail")
                except ValueError:
                    print("epbprtv0 fail")
            else:
                print("epbprtv0 fail")
    elif query_mode == QueryMode.PREPARED:
        # Prepared query mode
        parsed = None
        k = None
        for line in next_line():
            if not line:
                break
            elif query_parameters and line[0] == "query-params" \
                    and line[-1] == "set":
                _query_parameters(line)
            elif line == ["query"]:
                if parsed and k:
                    results = obj.query(parsed, k)
                    if results:
                        print("epbprtv0 ok %d" % len(results))
                        for index in results:
                            print("epbprtv0 %d" % index)
                    else:
                        print("epbprtv0 fail")
                else:
                    print("epbprtv0 fail")
            elif len(line) == 2:
                try:
                    parsed, k = parser(line[0]), int(line[1])
                    print("epbprtv0 ok prepared true")
                except ValueError:
                    print("epbprtv0 fail")
            else:
                print("epbprtv0 fail")
    elif query_mode == QueryMode.BATCH:
        # Batch query mode
        parsed = None
        k = None
        for line in next_line():
            if not line:
                break
            elif query_parameters and line[0] == "query-params" \
                    and line[-1] == "set":
                _query_parameters(line)
            elif line == ["query"]:
                if parsed and k:
                    results = obj.batch_query(parsed, k)
                    print("epbprtv0 ok")
                    for result in obj.get_batch_results():
                        if result:
                            print("epbprtv0 ok %d" % len(result))
                            for index in result:
                                print("epbprtv0 %d" % index)
                        else:
                            print("epbprtv0 fail")
                else:
                    print("epbprtv0 fail")
            elif len(line) > 1:
                try:
                    parsed, k = map(parser, line[0:-1]), int(line[-1])
                    print("epbprtv0 ok")
                except ValueError as e:
                    print("epbprtv0 fail" % e)
            else:
                print("epbprtv0 fail")
        pass
    print("epbprtv0 ok")
