''' Methods for profiling and timing sections of code '''

import cProfile
import pstats
import time

from datetime import timedelta
from io import StringIO


class Profiler():
    ''' Context manager for running a profile on a section of code

        Arguments:

            sortby:  Sorting for profile output, passed on to pstats.Stats instance
    '''

    def __init__(self, sortby='cumulative'):
        self._sortby = sortby
        self._profiler = cProfile.Profile()
        self._enabled = False

    def __enter__(self):
        if self._enabled:
            raise RuntimeError("Profiler already enabled")
        self._profiler.enable()
        self._enabled = True
        return self

    def __exit__(self, *exc):
        assert self._enabled
        self._profiler.disable()
        self._enabled = False

    def __str__(self):
        stream = StringIO()
        stats = pstats.Stats(self._profiler, stream=stream).sort_stats(self._sortby)
        stats.print_stats()
        return stream.getvalue()


class Timer():
    ''' Context manager for measuring the amount of time elapsed performing the work inside the context '''

    def __init__(self):
        self._start_seconds = None
        self._elapsed_time = None

    def __enter__(self):
        self._start_seconds = time.time()
        return self

    def __exit__(self, *exc):
        assert self._start_seconds is not None
        finish_seconds = time.time()
        self._elapsed_time = timedelta(seconds=finish_seconds - self._start_seconds)

    def __str__(self):
        return "Elapsed time: {}".format(self._elapsed_time)
