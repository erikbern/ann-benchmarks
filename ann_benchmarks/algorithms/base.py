from __future__ import absolute_import

class BaseANN(object):
    def use_threads(self):
        return True
    def done(self):
        pass
    def get_index_size(self, process):
        """Returns the size of the index in kB or -1 if not implemented."""
        try:
            statusfile = open("/proc/%(pid)s/status" % {"pid" : str(process)}, "r")
            for line in statusfile.readlines():
                if "VmRSS" in line:
                    mem_usage = line.split(":")[1].strip()
                    usage, unit = mem_usage.split(" ")
                    val = int(usage)
                    # Assume output to be in kB
                    if unit == "B":
                            val /= 1000.0
                    if unit == "mB":
                            val *= 1e3
                    if unit == "gB":
                            val *= 1e6
                    return val
        except:
            print("Couldn't open status file, no index size available.")
        return -1

    def fit(self, X):
        pass
    def query(self, q, k):
        return [] # array of candidate indices

    # def query_verbose(self, q, k):
    #     return (self.query(q, k), {}) # results with a dict of extra data
