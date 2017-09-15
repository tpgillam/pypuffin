from pypuffin.profile import Profiler

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

