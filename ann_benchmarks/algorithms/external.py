from __future__ import absolute_import
import shlex
import subprocess
from ann_benchmarks.algorithms.base import BaseANN

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
        if not self._program:
            self._program = subprocess.Popen(
                self._args,
                bufsize = 1, # line buffering
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                universal_newlines = True)
            for key, value in self._params.iteritems():
                self.__write("%s %s" % \
                    (Subprocess.__quote(key), Subprocess.__quote(value)))
                assert(self.__line()[0] == "ok")
            self.__write("")
            assert(self.__line()[0] == "ok")
        return self._program

    def __init__(self, args, encoder, params):
        self.name = "Subprocess(program = %s, %s)" % (args[0], str(params))
        self._program = None
        self._args = args
        self._encoder = encoder
        self._params = params

    def fit(self, X):
        for entry in X:
            self.__write(self._encoder(entry))
            assert(self.__line()[0] == "ok")
        self.__write("")
        assert(self.__line()[0] == "ok")

    def query(self, v, n):
        self.__write("%s %d" % \
            (Subprocess.__quote(self._encoder(v)), n))
        status = self.__line()
        if status[0] == "ok":
            count = int(status[1])
            results = []
            i = 0
            while i < count:
                line = self.__line()
                results.append(int(line[0]))
                i += 1
            assert(len(results) == count)
            return results
        else:
            assert(status[0] == "fail")
            return []

    def use_threads(self):
        return False
    def done(self):
        if self._program:
            self._program.terminate()
