import inspect

from functools import wraps
from inspect import Parameter


# TODO Do we want to create & expose a cached isinstance here to avoid MRO walking?

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
        ValueError: Argument 0 is 2.5, but should have type <class 'numbers.Integral'>

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
    '''
    def decorator(func):
        # Check that the signature matches.
        signature = inspect.signature(func)
        iter_args = iter(args)
        iter_kwargs = iter(kwargs.items())
        iter_parameters = iter(signature.parameters.items())
        for name, parameter in iter_parameters:
            try:
                # No checking to do for the positional arguments other than that the signature says that it should
                # be positional
                next(iter_args)
                if parameter.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                    print(parameter.kind)
                    raise ValueError("Mismatched signature")

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
                        raise ValueError("Default argument {} is {}, but should have type {}".format(name, 
                            parameter.default, val_kwarg))

                except StopIteration:
                    # Run out of args & kwargs. If *do* have any arguments left, then error.
                    if not _is_empty(iter_parameters):
                        raise ValueError("Mismatched signature")
        # If there are any elements left in iter_args or iter_kwargs, then we fail
        if not (_is_empty(iter_args) and _is_empty(iter_kwargs)):
            raise ValueError("Mismatched signature")


        @wraps(func)
        def new_func(*f_args, **f_kwargs):
            if len(args) != len(f_args):
                raise ValueError("Mismatched argument list")
            if len(f_kwargs.keys() - kwargs.keys()) > 0:
                raise ValueError("Mismatched keyword argument keys")

            for i, (arg, f_arg) in enumerate(zip(args, f_args)):
                if not _isinstance(f_arg, arg):
                    raise ValueError("Argument {} is {}, but should have type {}".format(i, f_arg, arg))

            for name in f_kwargs:
                arg = kwargs[name]
                f_arg = f_kwargs[name]
                if not _isinstance(f_arg, arg):
                    raise ValueError("Argument {} is {}, but should have type {}".format(name, f_arg, arg))

            return func(*f_args, **f_kwargs)
        return new_func
    return decorator

