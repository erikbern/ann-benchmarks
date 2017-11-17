import h5py
import os
import urllib.request

def get_dataset(which):
    hdf5_fn = os.path.join('data', '%s.hdf5' % which)
    if not os.path.exists(hdf5_fn):
        url = 'http://vectors.erikbern.com/%s' % hdf5_fn
        print('downloading %s...', url)
        urllib.request.urlretrieve(url, hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn)
    return hdf5_f

