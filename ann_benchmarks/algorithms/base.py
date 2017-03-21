from __future__ import absolute_import

class BaseANN(object):
    def use_threads(self):
        return True
    def done(self):
        pass
    def get_index_size(self):
	"""Returns the size of the index in kB or -1 if not implemented."""
	return -1
