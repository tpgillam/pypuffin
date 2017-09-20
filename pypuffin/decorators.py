''' Assorted decorators for adding commonly desired functionality '''

import inspect

from functools import wraps
from inspect import Parameter  # pylint: disable=ungrouped-imports

from pypuffin.contextlib import safecontextmanager


# Global state to monitor whether or not we want to disable accepts. We use a stack to support entering the context
# multiple times.
_TYPECHECKING_ENABLED = True
_TYPECHECKING_ENABLED_STACK = []


def typechecking_enabled():
    ''' Returns True iff typechecking is enabled '''
    return _TYPECHECKING_ENABLED


@safecontextmanager
def disable_typechecking():
    ''' Context manager to disable runtime type checking, for increased performance

        >>> print(1, typechecking_enabled())
        1 True
        >>> with disable_typechecking():
        ...     print(2, typechecking_enabled())
        ...     with disable_typechecking():
        ...         print(3, typechecking_enabled())
        ...     print(4, typechecking_enabled())
        2 False
        3 False
        4 False
        >>> print(5, typechecking_enabled())
        5 True
    '''
    global _TYPECHECKING_ENABLED  # pylint: disable=global-statement
    _TYPECHECKING_ENABLED_STACK.append(_TYPECHECKING_ENABLED)
    _TYPECHECKING_ENABLED = False
    yield
    _TYPECHECKING_ENABLED = _TYPECHECKING_ENABLED_STACK.pop()


def _isinstance(value, type_):
    ''' Re-implementation of isinstance with edge-cases, including supporting 'None'.

        >>> _isinstance(None, None)
        True
    '''
    # Special-casing for None
    if type_ is None:
        type_ = type(None)
    if isinstance(type_, tuple) and None in type_:
        type_ = tuple(type(None) if elem is None else elem for elem in type_)

    return isinstance(value, type_)


def _is_empty(iterator):
    ''' Return True iff the given iterator is empty. It can mutate the argument!

        >>> _is_empty(iter([]))
        True
        >>> _is_empty(iter([1]))
        False
    '''
    try:
        next(iterator)
    except StopIteration:
        return True
    else:
        return False


def accepts(*args, **kwargs):
    ''' Create a decorator to do type-checking on the arguments to the wrapped function

        >>> from numbers import Integral
        >>> @accepts(Integral)
        ... def foo(a):
        ...     return a ** 2
        >>> foo(2)
        4
        >>> foo(2.5)
        Traceback (most recent call last):
          ...
        ValueError: Argument a is 2.5, but should have type <class 'numbers.Integral'>

        >>> @accepts(Integral, b=(None, bool))
        ... def foo2(a, b=None):
        ...     b = b if b is not None else True
        ...     return a * (2 if b else 3)
        >>> foo2(2)
        4
        >>> foo2(2, b=None)
        4
        >>> foo2(2, b=3)
        Traceback (most recent call last):
          ...
        ValueError: Argument b is 3, but should have type (None, <class 'bool'>)

        >>> @accepts(Integral)
        ... def foo3():
        ...     return a
        Traceback (most recent call last):
          ...
        ValueError: Mismatched signature

        >>> @accepts(Integral, b=(None, bool))
        ... def foo4(a, b):
        ...     return a
        Traceback (most recent call last):
          ...
        ValueError: Mismatched signature

        >>> @accepts(Integral, b=(None, bool))
        ... def foo5(a):
        ...     return a
        Traceback (most recent call last):
          ...
        ValueError: Mismatched signature

        >>> @accepts(Integral, b=(None, bool))
        ... def foo6(a, b=3):
        ...     return a
        Traceback (most recent call last):
          ...
        ValueError: Default argument b is 3, but should have type (None, <class 'bool'>)

        >>> from numbers import Real
        >>> @accepts(Integral, b=Real)
        ... def foo7(a, b=0.3):
        ...     return a
        >>> foo7(2, b=0.1)
        2
        >>> foo7(2, 0.1)
        2

        >>> from pypuffin.types import Callable
        >>> @accepts(Integral, (None, Callable))
        ... def foo8(a, b=None):
        ...     if b is None:
        ...         return a
        ...     return b(a)
        >>> foo8(1)
        1
        >>> import math
        >>> foo8(4, math.sqrt)
        2.0
        >>> foo8(4, b=math.sqrt)
        2.0
        >>> foo8(4, 2)
        Traceback (most recent call last):
          ...
        ValueError: Argument b is 2, but should have type (None, <class 'pypuffin.types.Callable'>)
        >>> foo8(4, b=2)
        Traceback (most recent call last):
          ...
        ValueError: Argument b is 2, but should have type (None, <class 'pypuffin.types.Callable'>)

        Runtime typechecking can be disabled like so
        >>> with disable_typechecking():
        ...     foo8(4, b=2)
        Traceback (most recent call last):
          ...
        TypeError: 'int' object is not callable
    '''
    def decorator(func):  # pylint: disable=missing-docstring
        # Check that the signature matches.
        signature = inspect.signature(func)
        iter_args = iter(args)
        iter_kwargs = iter(kwargs.items())
        iter_parameters = iter(signature.parameters.items())

        # Store a mapping from argument name to expected type, use when doing call-time checks
        name_to_type = {}

        for name, parameter in iter_parameters:
            try:
                # No checking to do for the positional arguments other than that the signature says that it should
                # be positional
                val_arg = next(iter_args)
                if parameter.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                    print(parameter.kind)
                    raise ValueError("Mismatched signature")

                # Store positional argument
                name_to_type[name] = val_arg

            except StopIteration:
                try:
                    name_kwarg, val_kwarg = next(iter_kwargs)
                    if not name_kwarg == name:
                        raise ValueError("Mismatched signature")

                    # Check that this is a keyword argument, with a default
                    if parameter.kind != Parameter.POSITIONAL_OR_KEYWORD or parameter.default is Parameter.empty:
                        raise ValueError("Mismatched signature")

                    # Check that default arguments are all valid
                    if not _isinstance(parameter.default, val_kwarg):
                        raise ValueError("Default argument {} is {}, but should have type {}".format(
                            name, parameter.default, val_kwarg))

                    # Store keyword argument
                    name_to_type[name] = val_kwarg

                except StopIteration:
                    # Run out of args & kwargs. If *do* have any arguments left, then error.
                    if not _is_empty(iter_parameters):
                        raise ValueError("Mismatched signature")

        # If there are any elements left in iter_args or iter_kwargs, then we fail
        if not (_is_empty(iter_args) and _is_empty(iter_kwargs)):
            raise ValueError("Mismatched signature")

        @wraps(func)
        def new_func(*f_args, **f_kwargs):  # pylint: disable=missing-docstring
            if typechecking_enabled():
                bound_arguments = signature.bind(*f_args, **f_kwargs)
                for name, value in bound_arguments.arguments.items():
                    expected_type = name_to_type[name]
                    if not _isinstance(value, expected_type):
                        raise ValueError(f"Argument {name} is {value}, but should have type {expected_type}")
            return func(*f_args, **f_kwargs)
        return new_func
    return decorator
