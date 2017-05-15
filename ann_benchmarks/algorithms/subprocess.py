from __future__ import absolute_import
from os.path import basename
import shlex
import subprocess
from ann_benchmarks.data import \
    bit_unparse_entry, int_unparse_entry, float_unparse_entry
from ann_benchmarks.algorithms.base import BaseANN

class SubprocessStoppedError(Exception):
    def __init__(self, code):
        super(Exception, self).__init__(code)
        self.code = code

class Subprocess(BaseANN):
    def __raw_line(self):
        return shlex.split( \
            self.__get_program_handle().stdout.readline().strip())
    def __line(self):
        line = self.__raw_line()
        while len(line) < 1 or line[0] != "epbprtv0":
            line = self.__raw_line()
        return line[1:]

    @staticmethod
    def __quote(token):
        return "'" + str(token).replace("'", "'\\'") + "'"

    def __write(self, string):
        self.__get_program_handle().stdin.write(string + "\n")

    def __get_program_handle(self):
        if self._program:
            self._program.poll()
            if self._program.returncode:
                raise SubprocessStoppedError(self._program.returncode)
        else:
            self._program = subprocess.Popen(
                self._args,
                bufsize = 1, # line buffering
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                universal_newlines = True)

            for key, value in self._params.iteritems():
                self.__write("%s %s" % \
                    (Subprocess.__quote(key), Subprocess.__quote(value)))
                assert self.__line()[0] == "ok", """\
assigning value '%s' to option '%s' failed""" % (value, key)
            if self._prepared:
                self.__write("frontend prepared-queries 1")
                assert self.__line()[0] == "ok", """\
enabling prepared queries mode failed"""

            self.__write("")
            assert self.__line()[0] == "ok", """\
transitioning to training mode failed"""
        return self._program

    def __init__(self, args, encoder, params, prepared = False):
        self.name = "Subprocess(program = %s, %s)" % \
            (basename(args[0]), str(params))
        self._program = None
        self._args = args
        self._encoder = encoder
        self._params = params
        self._prepared = prepared
        if not prepared:
            self.query = self.__query_normal
        else:
            self.query = self.__query_prepared
        self._result_count = None

    def get_index_size(self, process = None):
	if not self._program:
		self.__get_program_handle()
	return super(Subprocess, self).get_index_size(str(self._program.pid))

    def supports_prepared_queries(self):
        return self._prepared

    def fit(self, X):
        for entry in X:
            d = Subprocess.__quote(self._encoder(entry))
            self.__write(d)
            assert self.__line()[0] == "ok", """\
encoded training point '%s' was rejected""" % d
        self.__write("")
        assert self.__line()[0] == "ok", """\
transitioning to query mode failed"""

    def __query_normal(self, v, n):
        d = Subprocess.__quote(self._encoder(v))
        self.__write("%s %d" % (d, n))
        return self.__handle_query_response()

    def __query_prepared(self, v, n):
        self.prepare_query(v, n)
        return self.run_prepared_query()

    def __handle_query_response(self):
        status = self.__line()
        if status[0] == "ok":
            count = int(status[1])
            results = []
            i = 0
            while i < count:
                line = self.__line()
                results.append(int(line[0]))
                i += 1
            assert len(results) == count
            return results
        else:
            assert status[0] == "fail", """\
query neither succeeded nor failed"""
            return []

    def prepare_query(self, v, n):
        d = Subprocess.__quote(self._encoder(v))
        self.__write("%s %d" % (d, n))
        assert self.__line()[0] == "ok", """\
preparing the query '%s' failed""" % d

    def run_prepared_query(self):
        self.__write("query")
        status = self.__line()
        if status[0] == "ok":
            self._result_count = int(status[1])
        else:
            assert status[0] == "fail", """\
query neither succeeded nor failed"""
            self._result_count = 0

    def get_prepared_query_results(self):
        results = []
        i = 0
        while i < self._result_count:
            line = self.__line()
            results.append(int(line[0]))
            i += 1
        self._result_count = 0
        return results

    def use_threads(self):
        return False
    def done(self):
        if self._program:
            self._program.poll()
            if not self._program.returncode:
                self._program.terminate()

def BitSubprocess(args, params):
    return Subprocess(args, bit_unparse_entry, params, False)

def BitSubprocessPrepared(args, params):
    return Subprocess(args, bit_unparse_entry, params, True)

def FloatSubprocess(args, params):
    return Subprocess(args, float_unparse_entry, params, True)

def IntSubprocess(args, params):
    return Subprocess(args, int_unparse_entry, params, True)
