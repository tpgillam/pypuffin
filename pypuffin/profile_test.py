''' Tests for pypuffin.profile '''

from pypuffin.profile import Profiler, Timer

from unittest import TestCase


class TestProfile(TestCase):
    ''' Tests for the profile module '''

    def test_profiler(self):
        ''' Test for the Profiler class '''
        with Profiler() as profiler:
            for _ in range(1000):
                pass
        result = str(profiler)
        self.assertIn('2 function calls in', result)
        self.assertIn('Ordered by: cumulative time', result)
        self.assertIn('ncalls  tottime  percall  cumtime  percall filename:lineno(function)', result)

    def test_timer(self):
        ''' Test for the Timer class '''
        with Timer() as timer:
            for _ in range(1000):
                pass
        result = str(timer)
        self.assertEqual(len(result.splitlines()), 1)
        self.assertIn('Elapsed time', result)
