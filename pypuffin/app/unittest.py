''' Run the unittests '''

import nose
import sys

from nose.plugins import Plugin
from nose.plugins.doctests import DocTestCase


class IgnoreDocstrings(Plugin):
    ''' Don't use docstrings when printing out test names. Adapted from:

        https://github.com/schlamar/nose-ignore-docstring/blob/master/nose_ignoredoc.py
    '''

    name = 'ignore-docstrings'

    def describeTest(self, test):
        ''' Summarise the test, for verbose output '''
        is_doctest = isinstance(test.test, DocTestCase)
        if is_doctest:
            # Returning None is equivalent to not implementing this case. We are happy with the
            # default rendering here.
            return
        return str(test)


def main():
    if len(sys.argv) == 0:
        modules = ['pypuffin']
    else:
        modules = sys.argv
    nose.run(addplugins=[IgnoreDocstrings()],
             argv=modules + ['--with-doctest', '-v', '--with-ignore-docstrings'])


if __name__ == '__main__':
    main()
