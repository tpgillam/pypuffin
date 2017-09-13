from types import MethodType


def instance_to_string(obj, include_private=False, include_type_properties=False, include_methods=False):
    ''' Return a reasonable string representation of the given object, including any public properties. By default
        omit private properties, and things that are present on the type, since they are likely not interesting. We
        also skip methods by default.
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
            '\n'.join('  {}: {},'.format(key, value) for key, value in sorted(key_to_value.items())))
