from __future__ import absolute_import
from os.path import basename
import shlex
from types import MethodType
import psutil
import subprocess
from ann_benchmarks.data import \
    bit_unparse_entry, int_unparse_entry, float_unparse_entry
from ann_benchmarks.algorithms.base import BaseANN


class SubprocessStoppedError(Exception):
    def __init__(self, code):
        super(Exception, self).__init__(code)
        self.code = code


class Subprocess(BaseANN):
    def _raw_line(self):
        return shlex.split(
            self._get_program_handle().stdout.readline().strip())

    def _line(self):
        line = self._raw_line()
#       print("<- %s" % (" ".join(line)))
        while len(line) < 1 or line[0] != "epbprtv0":
            line = self._raw_line()
        return line[1:]

    @staticmethod
    def _quote(token):
        return "'" + str(token).replace("'", "'\\'") + "'"

    def _write(self, string):
        #       print("-> %s" % string)
        self._get_program_handle().stdin.write(string + "\n")

    # Called immediately before transitioning from query mode to training mode
    def _configuration_hook(self):
        pass

    def _get_program_handle(self):
        if self._program:
            self._program.poll()
            if self._program.returncode:
                raise SubprocessStoppedError(self._program.returncode)
        else:
            self._program = subprocess.Popen(
                self._args,
                bufsize=1,  # line buffering
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True)

            for key, value in iter(self._params.items()):
                self._write("%s %s" %
                            (Subprocess._quote(key), Subprocess._quote(value)))
                assert self._line()[0] == "ok", """\
assigning value '%s' to option '%s' failed""" % (value, key)
            self._configuration_hook()

            self._write("")
            assert self._line()[0] == "ok", """\
transitioning to training mode failed"""
        return self._program

    def __init__(self, args, encoder, params):
        self.name = "Subprocess(program = %s, %s)" % \
            (basename(args[0]), str(params))
        self._program = None
        self._args = args
        self._encoder = encoder
        self._params = params

    def get_memory_usage(self):
        if not self._program:
            self._get_program_handle()
        return psutil.Process(pid=self._program.pid).memory_info().rss / 1024

    def fit(self, X):
        for entry in X:
            d = Subprocess._quote(self._encoder(entry))
            self._write(d)
            assert self._line()[0] == "ok", """\
encoded training point '%s' was rejected""" % d
        self._write("")
        assert self._line()[0] == "ok", """\
transitioning to query mode failed"""

    def query(self, v, n):
        d = Subprocess._quote(self._encoder(v))
        self._write("%s %d" % (d, n))
        return self._handle_query_response()

    def _handle_query_response(self):
        status = self._line()
        if status[0] == "ok":
            count = int(status[1])
            return self._collect_query_response_lines(count)
        else:
            assert status[0] == "fail", """\
query neither succeeded nor failed"""
            return []

    def _collect_query_response_lines(self, count):
        results = []
        i = 0
        while i < count:
            line = self._line()
            results.append(int(line[0]))
            i += 1
        return results

    def done(self):
        if self._program:
            self._program.poll()
            if not self._program.returncode:
                self._program.terminate()


class PreparedSubprocess(Subprocess):
    def __init__(self, args, encoder, params):
        super(PreparedSubprocess, self).__init__(args, encoder, params)
        self._result_count = None

    def _configuration_hook(self):
        self._write("frontend prepared-queries 1")
        assert self._line()[0] == "ok", """\
enabling prepared queries mode failed"""

    def query(self, v, n):
        self.prepare_query(v, n)
        self.run_prepared_query()
        return self.get_prepared_query_results()

    def prepare_query(self, v, n):
        d = Subprocess._quote(self._encoder(v))
        self._write("%s %d" % (d, n))
        assert self._line()[0] == "ok", """\
preparing the query '%s' failed""" % d

    def run_prepared_query(self):
        self._write("query")
        status = self._line()
        if status[0] == "ok":
            self._result_count = int(status[1])
        else:
            assert status[0] == "fail", """\
query neither succeeded nor failed"""
            self._result_count = 0

    def get_prepared_query_results(self):
        if self._result_count:
            try:
                return self._collect_query_response_lines(self._result_count)
            finally:
                self._result_count = 0
        else:
            return []


class BatchSubprocess(Subprocess):
    def __init__(self, args, encoder, params):
        super(BatchSubprocess, self).__init__(args, encoder, params)
        self._qp_count = None

    def _configuration_hook(self):
        self._write("frontend batch-queries 1")
        assert self._line()[0] == "ok", """\
enabling batch queries mode failed"""

    def query(self, v, n):
        self.prepare_batch_query([v], n)
        self.run_batch_query()
        return self.get_batch_results()[0]

    def prepare_batch_query(self, X, n):
        d = " ".join(map(lambda p: Subprocess._quote(self._encoder(p)), X))
        self._qp_count = len(X)
        self._write("%s %d" % (d, n))
        assert self._line()[0] == "ok", """\
preparing the batch query '%s' failed""" % d

    def run_batch_query(self):
        self._write("query")
        status = self._line()
        assert status[0] == "ok", """\
batch query failed completely"""

    def get_batch_results(self):
        results = []
        i = 0
        while i < self._qp_count:
            #           print("%d/%d" % (i, self._qp_count))
            status = self._line()
            if status[0] == "ok":
                rc = int(status[1])
                results.append(self._collect_query_response_lines(rc))
            else:
                results.append([])
            i += 1
        return results


def BitSubprocess(args, params):
    return Subprocess(args, bit_unparse_entry, params)


def BitSubprocessPrepared(args, params):
    return PreparedSubprocess(args, bit_unparse_entry, params)


def FloatSubprocess(args, params):
    return Subprocess(args, float_unparse_entry, params)


def FloatSubprocessPrepared(args, params):
    return PreparedSubprocess(args, float_unparse_entry, params)


def FloatSubprocessBatch(args, params):
    return BatchSubprocess(args, float_unparse_entry, params)


def IntSubprocess(args, params):
    return Subprocess(args, int_unparse_entry, params)


def QueryParamWrapper(constructor, args, params):
    r = constructor(args, params)

    def _do(self, original=r._configuration_hook):
        original()
        self._write("frontend query-parameters 1")
        assert self._line()[0] == "ok", """\
enabling query parameter support failed"""
    r._configuration_hook = MethodType(_do, r)

    def _sqa(self, *args):
        self._write("query-params %s set" %
                    (" ".join(map(Subprocess._quote, args))))
        assert self._line()[0] == "ok", """\
reconfiguring query parameters failed"""
        print(args)
    r.set_query_arguments = MethodType(_sqa, r)
    return r
