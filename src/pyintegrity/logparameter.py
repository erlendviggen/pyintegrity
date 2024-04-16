from typing import Any
from dlisio import dlis
from pyintegrity import Quantity
from .helpers import to_quantity


class LogParameter:
    """Log parameter container class

    Each `LogParameter` has a name (aka mnemonic), a description (aka long name), and a set of values. The number of
    values per parameter is arbitrary; for 0 values, we store `None`, for 1 value, we store the value itself, and for
    more values, we store them in an array. Values can have several types, but most are numbers with units (stored as
    `Quantity` objects) or strings.

    Args:
        parameter: DLIS parameter to extract data from

    Attributes:
        name (str | None): Name of the log parameter, typically a DLIS mnemonic
        description (str | None): Description of the log parameter, typically from a DLIS file
        values (Any | list[Any]): The parameter value(s)

    """
    def __init__(self, parameter: dlis.Parameter) -> None:
        self.name = parameter.name
        self.description = parameter.long_name
        if len(parameter.values) == 0:
            self.values = None
        elif parameter.values.dtype.kind in ('S', 'U'):   # Parameter values are strings
            self.values = parameter.values[0] if len(parameter.values) == 1 else parameter.values
        else:   # Parameter values are numbers, possibly with units
            self.values = to_quantity(parameter)

    def __repr__(self) -> str:
        if isinstance(self.values, Quantity):
            try:
                return f'{self.name} = {self.values:~P} [{self.description}]'
            except KeyError:   # Format ~P fails for e.g. units dB/m
                return f'{self.name} = {self.values:P} [{self.description}]'
        else:
            return f'{self.name} = {self.values} [{self.description}]'
