import json
import os
import subprocess

print('Building base image...')
subprocess.check_call('docker build --rm -t ann-benchmarks -f install/Dockerfile .', shell=True)

def build(library):
    print('Building %s...' % library)
    subprocess.check_call('docker build --rm -t ann-benchmarks-%s -f install/Dockerfile.%s .' % (library, library), shell=True)

if os.getenv('LIBRARY'):
    build(os.getenv('LIBRARY'))
else:
    for fn in os.listdir('install'):
        if fn.startswith('Dockerfile.'):
            build(fn.split('.')[-1])
