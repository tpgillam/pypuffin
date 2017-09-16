''' Iterators, building on those in e.g. itertools '''

from itertools import tee


def pairs(iterable, skip=1):
    ''' Given an iterable s of size N, return an iterator to the following sequence.

            (s[0], s[skip]), (s[1], s[skip + 1]), ..., (s[N - 1 - skip], s[N - 1])

        >>> list(pairs([1, 2, 3]))
        [(1, 2), (2, 3)]

        >>> list(pairs([1, 2, 3, 4, 5], skip=2))
        [(1, 3), (2, 4), (3, 5)]
    '''
    a, b = tee(iterable)
    for _ in range(skip):
        next(b, None)
    return zip(a, b)
