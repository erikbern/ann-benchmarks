import h5py
import os

def get_dataset(which):
    hdf5_fn = os.path.join('data', '%s.hdf5' % which)
    hdf5_f = h5py.File(hdf5_fn)
    return hdf5_f

