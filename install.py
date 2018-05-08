import docker
import json
import os
import sys

client = docker.from_env()
print('Building base image...')
client.images.build(path=os.getcwd(), tag='ann-benchmarks', rm=True, dockerfile='install/Dockerfile')

def build(library):
    print('Building %s...' % library)
    try:
        client.images.build(path=os.getcwd(), tag='ann-benchmarks-%s' % library, rm=True, dockerfile='install/Dockerfile.%s' % library)
    except docker.errors.BuildError as err:
        raise Exception("Build error: {0}".format(err))

if os.getenv('LIBRARY'):
    build(os.getenv('LIBRARY'))
else:
    for fn in os.listdir('install'):
        if fn.startswith('Dockerfile.'):
            build(fn.split('.')[-1])
