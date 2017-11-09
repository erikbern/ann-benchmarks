#!/usr/bin/env python

from ann_benchmarks.results import enumerate_result_files

counts = {}

for descriptor, _ in enumerate_result_files():
  ds = descriptor["dataset"]
  counts[ds] = counts[ds] + 1 if ds in counts else 1

if counts:
  print "Experiments run for each dataset:"
  total = 0
  for ds in counts:
    print "%s: %d" % (ds, counts[ds])
    total += counts[ds]
  print "--"
  print "total %d" % total
else:
  print "No experiments run yet"
