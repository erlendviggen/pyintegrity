import numpy as np
from pyintegrity import Quantity
from .logchannel import LogChannel
from .helpers import to_quantity


class Material:
    """Contains information on the parameters of a material

    1-3 material parameters may be provided. If 2 are provided, the 3rd parameter can be calculated.

    Args:
        speed: P-wave speed of the material
        impedance: P-wave impedance of the material
        density: Mass density of the material
    """
    def __init__(self,
                 speed: float | np.ndarray | Quantity | LogChannel | None = None,
                 impedance: float | np.ndarray | Quantity | LogChannel | None = None,
                 density: float | np.ndarray | Quantity | LogChannel | None = None) -> None:
        if speed is not None:
            speed = to_quantity(speed, 'm/s')
        self._speed = speed
        if impedance is not None:
            impedance = to_quantity(impedance, 'MRayl')
        self._impedance = impedance
        if density is not None:
            density = to_quantity(density, 'kg/m^3')
        self._density = density

    @property
    def speed(self) -> Quantity | None:
        """Get the P-wave speed in m/s, calculating if necessary"""
        if self._speed is not None:
            return self._speed
        if self._impedance is not None and self._density is not None:
            speed = self._impedance / self._density
            return speed.to('m/s')
        return None
        
    @property
    def impedance(self) -> Quantity | None:
        """Get the P-wave impedance in MRayl, calculating if necessary"""
        if self._impedance is not None:
            return self._impedance
        if self._speed is not None and self._density is not None:
            impedance = self._speed * self._density
            return impedance.to('MRayl')
        return None
    
    @property
    def density(self) -> Quantity | None:
        """Get the density in kg/m^3, calculating if necessary"""
        if self._density is not None:
            return self._density
        if self._speed is not None and self._impedance is not None:
            density = self._impedance / self._speed
            return density.to('kg/m^3')
        return None

    def get_depth_section(self, i_start: int, i_end: int) -> 'Material':
        """Return a `Material` copy which only contains data in the specified range

        Args:
            i_start: Start index of the range
            i_end: End index of the range

        Returns:
            A new `Material` containing the data in the specified range
        """
        speed = None if self._speed is None else self._speed[i_start:i_end]
        impedance = None if self._impedance is None else self._impedance[i_start:i_end]
        density = None if self._density is None else self._density[i_start:i_end]
        return Material(speed=speed, impedance=impedance, density=density)
