''' Methods for creating and manipulating strings '''

from types import MethodType


def instance_to_string(obj, include_private=False, include_type_properties=False, include_methods=False):
    ''' Return a reasonable string representation of the given object, including any public properties. By default
        omit private properties, and things that are present on the type, since they are likely not interesting. We
        also skip methods by default.

        >>> class Moo():
        ...     def __init__(self):
        ...         self.a = 1
        ...         self.b = '42'
        ...         self._c = 3.14
        >>> moo = Moo()
        >>> print(instance_to_string(moo))
        Moo<
          a: 1,
          b: '42',
        >
        >>> print(instance_to_string(moo, include_private=True))
        Moo<
          _c: 3.14,
          a: 1,
          b: '42',
        >
    '''
    type_keys = set(dir(type(obj)))
    is_valid = lambda key, value: (
        not key.startswith('__') and
        (include_private or not key.startswith('_')) and
        (include_type_properties or key not in type_keys) and
        (include_methods or not isinstance(value, MethodType))
        )

    key_to_value = {key: value for key, value in obj.__dict__.items() if is_valid(key, value)}
    for key in dir(obj):
        value = getattr(obj, key)
        if not is_valid(key, value):
            continue
        key_to_value[key] = value

    return '{}<\n{}\n>'.format(
        type(obj).__name__,
        '\n'.join('  {}: {!r},'.format(key, value) for key, value in sorted(key_to_value.items())))
