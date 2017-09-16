from pypuffin.profile import Profiler, Timer

from unittest import TestCase


class TestProfile(TestCase):
    ''' Tests for the profile module '''

    def test_profiler(self):
        ''' Test for the Profiler class '''
        with Profiler() as p:
            for _ in range(1000):
                pass
        result = str(p)
        self.assertIn('2 function calls in', result)
        self.assertIn('Ordered by: cumulative time', result)
        self.assertIn('ncalls  tottime  percall  cumtime  percall filename:lineno(function)', result)

    def test_timer(self):
        ''' Test for the Timer class '''
        with Timer() as t:
            for _ in range(1000):
                pass
        result = str(t)
        self.assertEqual(len(result.splitlines()), 1)
        self.assertIn('Elapsed time', result)

