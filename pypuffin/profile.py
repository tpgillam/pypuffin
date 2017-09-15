''' Methods for profiling and timing sections of code '''

import cProfile
import pstats

from io import StringIO


class Profiler():
    ''' Context manager for running a profile on a section of code

        Arguments:

            sortby:  Sorting for profile output, passed on to pstats.Stats instance
    '''

    def __init__(self, sortby='cumulative'):
        self._sortby = sortby
        self._profiler  = cProfile.Profile()
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

