from typing import Union, TYPE_CHECKING
import numpy as np
from dlisio import dlis
from pyintegrity import Quantity, Unit


if TYPE_CHECKING:   # Needed to avoid a circular import problem; True during static type checking, False during runtime
    from .logchannel import LogChannel


# Need to use string 'LogChannel' to avoid circular import. Union[...] avoids a problem with type | 'type'.
def to_quantity(value: Union[float, np.ndarray, Quantity, dlis.Parameter, 'LogChannel'], 
                target_units: str | Unit | None = None) -> Quantity:
    """Returns the input value as a pint Quantity with a particular set of units

    If the input value is a number or an array, it is assumed to already be in the target units, which are applied
    directly to return a `Quantity`. If the input value is already a `Quantity`, the value are converted to the target
    units. Unlike basic pint unit conversions, this function also supports reciprocal unit conversion, for example when
    converting from slownesses in e.g. µs/ft to speeds in e.g. m/s: `to_quantity(Quantity(203.2, 'µs/ft'), 'm/s')`.

    Args:
        value: The input value to be assigned the target units
        target_units: The target units to be assigned to the input value. `None` implies no unit conversion.

    Raises:
        RuntimeError: If an input `Quantity` cannot be converted to the target units

    Returns:
        A `Quantity` object with the target units
    """
    from .logchannel import LogChannel  #pylint: disable=import-outside-toplevel; done to avoid a circular import

    if isinstance(value, dlis.Parameter):   # Must convert to a Quantity with the original units
        number = value.values[0] if len(value.values) == 1 else value.values
        units = value.attic['VALUES'].units
        if units:
            try:
                value = Quantity(number, units)
            except ValueError:
                value = number * Quantity(units)   # To handle units like "0.1 in"
        else:
            value = Quantity(number)
    elif isinstance(value, LogChannel):   # Must convert to a Quantity with the original units
        value = value.array

    if isinstance(value, Quantity):
        if target_units is None:
            return value
        # Convert to the target units, taking the reciprocal if necessary
        if value.units.is_compatible_with(target_units):
            return value.to(target_units)   # type: ignore[return-value]
        if (1/value.units).is_compatible_with(target_units):
            return (1/value).to(target_units)
        raise RuntimeError(f'Cannot convert Quantity with units {value.units} to units {target_units}')
    
    return Quantity(value, target_units)
