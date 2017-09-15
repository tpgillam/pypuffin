from unittest import TestCase

from pypuffin.types import Callable


class TestTypes(TestCase):
    ''' Tests for the types module '''

    def test_callable(self):
        ''' Test Callable '''
        self.assertIsInstance(print, Callable)
        self.assertNotIsInstance(object(), Callable)
        self.assertNotIsInstance(1, Callable)
        self.assertNotIsInstance('a', Callable)
        class Moo():
            def __call__(self):
                pass
        self.assertIsInstance(Moo(), Callable)

