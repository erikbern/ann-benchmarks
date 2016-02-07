import struct
import sys
import gzip

f = gzip.open(sys.argv[1])
g = gzip.open(sys.argv[2], 'w')
words, size = (int(x) for x in f.readline().strip().split())

t = 'f' * size

while True:
    while True:
        ch = f.read(1)
        if ch in ['', '\n', ' ']: break
    if ch in ['', '\n']: break
    vec = struct.unpack(t, f.read(4 * size))

    print >> g, ' '.join(['%.6f' % x for x in vec])
g.close()
