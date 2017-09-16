''' Some custom types and type-like objects '''


class _CallableMeta(type):
    ''' Metaclass for Callable '''

    def __instancecheck__(cls, instance):
        return callable(instance)


class Callable(metaclass=_CallableMeta):
    ''' Type that appears to be an instance of any object that is callable '''

    def __init__(self):
        raise RuntimeError("Do not instantiate Callable")
