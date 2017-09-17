''' Extensions of the contextlib library '''

from contextlib import contextmanager
from functools import wraps


def safecontextmanager(func):
    ''' Behaves similarly to context manager, but if an exception occurs during the context execution, we ensure
        that the cleanup code is called prior to re-raising the exception. This makes it easier to follow the
        recommendation in the contextlib documentation, which is to wrap the 'yield' in a try: block.

        A dangerous example with contextlib...

        >>> x = 0
        >>> @contextmanager
        ... def unsafe_foo():
        ...     global x
        ...     x = 1
        ...     yield
        ...     x = 3
        >>> with unsafe_foo():
        ...     x = 2
        >>> x
        3
        >>> with unsafe_foo():
        ...     x = 2
        ...     raise RuntimeError
        Traceback (most recent call last):
          ...
        RuntimeError
        >>> x
        2

        And now for the safe version...

        >>> x = 0
        >>> @safecontextmanager
        ... def safe_foo():
        ...     global x
        ...     x = 1
        ...     yield
        ...     x = 3
        >>> with safe_foo():
        ...     x = 2
        >>> x
        3
        >>> with safe_foo():
        ...     x = 2
        ...     raise RuntimeError
        Traceback (most recent call last):
          ...
        RuntimeError
        >>> x
        3
    '''
    @wraps(func)
    def wrapped_func(*args, **kwargs):  # pylint: disable=missing-docstring
        generator = func(*args, **kwargs)
        try:
            yield next(generator)
        except Exception as exc:  # pylint: disable=broad-except
            try:
                next(generator)
            except StopIteration:
                raise exc
        else:
            next(generator)

    # Now that the function has been made safe, wrap it up as a context manager
    return contextmanager(wrapped_func)
