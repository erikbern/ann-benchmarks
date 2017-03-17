#!/usr/bin/env python

# (Taken from data/data_conv/convert_texmex_fvec.py in the nmslib project,
# made available under the Apache 2.0 licence.)

# Converting fvec-files from the texmex collection:
# http://corpus-texmex.irisa.fr/
# Should be run on an Intel (little-endian) machine

import sys
import struct
import array
import os

ifile = sys.argv[1]
maxQty = 0
if len(sys.argv) == 3:
  maxQty = int(sys.argv[2])

fsize = os.path.getsize(ifile)

fin = open(ifile, 'rb')

dim = struct.unpack('i', fin.read(4))[0]

if fsize % (4*(dim + 1)) != 0:
  raise Exception("File: " + ifile + " has the wrong format, expected dim = " + str(dim))

fin.close()


fin = open(ifile, 'rb')

rowQty = fsize / (dim + 1) / 4

#print rowQty, dim

if maxQty != 0:
  rowQty = min(rowQty, maxQty)

for i in range(0,rowQty):
  tmpd = struct.unpack('<i', fin.read(4))[0]
  if tmpd != dim:
    raise Exception("Wrong format, got wrong dimension: " + str(tmpd) + " expect: " + str(dim) + " row " + str(i+1))

  vec = array.array('f')
  
  vec.read(fin, dim)
  out = ''
  for j in range(0,dim):
    if j > 0: out = out + ' '
    out = out + str(vec[j])
  print out
    
