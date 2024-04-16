from typing import Any, Self
import copy
import numpy as np
import matplotlib.pyplot as plt
from pyintegrity import Quantity, Unit, xr
from .helpers import to_quantity


class LogChannel:
    """Multidimensional log channel data

    The log channel can have 1-3 dimensions. The first is always depth (`z`), and the two others can be angle (`phi`)
    and either time (`t`) or frequency (`f`). The mandatory order is depth-angle-time/frequency. Thus, `LogChannel` can
    store depth-only channels (e.g., CBL), depth-angle channels (e.g., outer impedance maps), depth-time channels (e.g.,
    VDL), depth-angle-time channels (e.g., pulse-echo waveforms), and depth-angle-frequency channels (e.g., pulse-echo
    waveforms frequency spectra).

    Internally, the log channel is stored as an `xarray.DataArray`, made unit-aware by the pint-xarray package. Because
    getting the axes from such `DataArray` objects is not entirely straightforward, the `LogChannel` defines properties
    to get the depth (`z`), angle (`phi`), time (`t`), or frequency (`f`) axis. The underlying array of the `DataArray`
    is always a `pint.Quantity`, whether the log channel has units or not, and can be accessed through the `array`
    property.

    Args:
        data: The underlying array; dimensions must be ordered as depth-angle-time
        z: The required depth axis of the log channel
        phi: The optional angle axis of the log channel
        t: The optional time axis of the log channel
        f: The optional frequency axis of the log channel
    
    Attributes:
        name (str | None): Name of the log channel, typically a DLIS mnemonic
        description (str | None): Description of the log channel, typically from a DLIS file
        data (xr.DataArray): The underlying data array as an unit-aware `xarray.DataArray`

    Raises:
        ValueError: Issues with the data array dimension order or the data array shape not matching the axes
    """
    def __init__(self,
                 data: np.ndarray | Quantity,
                 z: np.ndarray | Quantity,
                 phi: np.ndarray | Quantity | None = None,
                 t: np.ndarray | Quantity | None = None,
                 f: np.ndarray | Quantity | None = None,
                 name: str | None = None,
                 description: str | None = None) -> None:
        self.name = name
        self.description = description

        if t is not None and f is not None:
            raise ValueError('LogChannel got both time and frequency axes')

        # Ensure that the axes have the correct units
        z = to_quantity(np.atleast_1d(z), 'm')
        if phi is not None and not isinstance(phi, Quantity):
            phi = to_quantity(np.atleast_1d(phi), 'deg')
        if t is not None and not isinstance(t, Quantity):
            t = to_quantity(np.atleast_1d(t), 's')
        if f is not None and not isinstance(f, Quantity):
            f = to_quantity(np.atleast_1d(f), 'Hz')
        
        # Ensure that the data array has the right size
        n_axes = (z is not None) + (phi is not None) + (t is not None) + (f is not None)
        if n_axes < data.ndim:
            raise ValueError(f'LogChannel got a {data.ndim}-dimensional data array but only {n_axes} axes')
        data = self._array_to_nd(np.atleast_1d(data), target_ndim=n_axes)

        # Ensure that the data array is ordered as depth, angle, time/frequency
        i_dim = 0
        for axis in (z, phi, t, f):
            if axis is None:
                continue
            if data.shape[i_dim] == len(axis):
                i_dim += 1
                continue
            raise ValueError(f'data shape {data.shape} does not match input axes in depth-angle-time/freq order')
        if i_dim != data.ndim:
            raise ValueError(f'data shape {data.shape} does not match input axes in depth-angle-time/freq order')

        # Build DataArray
        coords: dict[str, Quantity] = {name: ax for name, ax in [('z', z), ('phi', phi), ('t', t), ('f', f)]
                                       if ax is not None}
        data_unitless = xr.DataArray(Quantity(data), coords={name: ax.magnitude for name, ax in coords.items()})
        data_unitless = self._ensure_increasing_coords(data_unitless)
        self.data: xr.DataArray = data_unitless.pint.quantify({name: ax.units for name, ax in coords.items()})

    @property
    def z(self) -> Quantity:
        """Get depth axis as a Quantity"""
        return Quantity(self.data.z.to_numpy(), self.data.z.attrs['units'])
    
    @property
    def phi(self) -> Quantity | None:
        """Get angle axis as a Quantity, if it exists"""
        if 'phi' not in self.data.coords:
            return None
        return Quantity(self.data.phi.to_numpy(), self.data.phi.attrs['units'])

    @property
    def t(self) -> Quantity | None:
        """Get time axis as a Quantity, if it exists"""
        if 't' not in self.data.coords:
            return None
        return Quantity(self.data.t.to_numpy(), self.data.t.attrs['units'])
    
    @property
    def f(self) -> Quantity | None:
        """Get frequency axis as a Quantity, if it exists"""
        if 'f' not in self.data.coords:
            return None
        return Quantity(self.data.f.to_numpy(), self.data.f.attrs['units'])
    
    @property
    def array(self) -> Quantity:
        """Get the underlying data array"""
        return self.data.data
    
    def to(self, units: str | Unit) -> Self:
        """Convert channel units
        
        Args:
            units: Compatible units to convert to
        """
        out = self.copy()
        out.data = out.data.pint.to(units)
        return out

    def to_numpy(self) -> np.ndarray:
        """Get the underlying data array as a unitless numpy array"""
        return self.data.to_numpy()

    def interpolate_to(self,
                       z: Quantity | None = None,
                       phi: Quantity | None = None,
                       t: Quantity | None = None,
                       f: Quantity | None = None,
                       extrapolate: bool = True,
                       method: str = 'linear') -> Self:
        """Interpolate the log channel to a new set of axes

        Args:
            z: Optional new depth axis to interpolate to
            phi: Optional new angle axis to interpolate to
            t: Optional new time axis to interpolate to
            f: Optional new frequency axis to interpolate to
            extrapolate: When going outside the original axes, whether to fill by edge values (`True`) or NaNs (`False`)
            method: Interpolation method to use

        Returns:
            A new LogChannel with the interpolated data
        """
        kwargs: dict[str, Any] = {'fill_value': 'extrapolate'} if extrapolate else {'fill_value': np.nan}
        coords = {name: ax for name, ax in [('z', z), ('phi', phi), ('t', t), ('f', f)] if ax is not None}
        
        new = copy.copy(self)
        new.data = new.data.pint.interp(coords, method=method, kwargs=kwargs)
        return new

    def plot_in(self,
                ax: plt.Axes,
                z: Quantity | None = None,
                phi: Quantity | None = None,
                t: Quantity | None = None,
                f: Quantity | None = None,
                method: str = 'nearest',
                **kwargs) -> plt.Axes:
        """Plot log channel data

        The arguments `z`, `phi`, `t`, and `f` can be used to reduce the dimensionality of the data before plotting or
        to select a subset for plotting.

        Args:
            ax: matplotlib `Axes` to plot into
            z: Optional depths to select before plotting
            phi: Optional angles to select before plotting
            t: Optional times to select before plotting
            f: Optional frequencies to select before plotting
            method: Selection method

        Returns:
            Whatever `xarray.DataArray.plot` returns
        """
        coords = {name: ax for name, ax in [('z', z), ('phi', phi), ('t', t), ('f', f)] if ax is not None}
        data: xr.DataArray = self.data.pint.sel(coords, method=method)
        
        # Specify some reasonable plot defaults
        if 'z' in data.coords.keys() and data.coords['z'].size > 1:   # The data is indexed by depth
            if 'yincrease' not in kwargs:
                kwargs['yincrease'] = False   # Make sure depth increases downwards
            if len(data.coords) == 1 and 'y' not in kwargs:
                kwargs['y'] = 'z'   # 1D line plot; make sure depth is on the y axis
        
        return data.plot(ax=ax, **kwargs)

    def copy(self) -> Self:
        """Make a shallow copy of the object with a deep copy of the data array"""
        new = copy.copy(self)
        new.data = self.data.copy()
        return new

    def get_depth_section(self, top: float, bottom: float) -> Self:
        """Return a copy which only contains data in the specified interval of the depth axis

        Args:
            top: Start of depth interval, in m
            bottom: End of depth interval, in m

        Returns:
            A copy of this `LogChannel` containing only an interval of the original depth axis
        """
        channel = self.copy()
        channel.data = channel.data.sel(z=slice(top, bottom))
        return channel

    def equalize_by_percentiles(self,
                              reference: 'LogChannel',
                              percentiles: tuple[float, float] = (25, 75)) -> Self:
        """Return a copy which has been equalized with a reference according to the values at the specified percentiles

        Args:
            reference: Reference `LogChannel` to equalize with
            percentiles: Percentiles to equalize values at

        Returns:
            A new and equalized `LogChannel`
        """
        assert len(percentiles) == 2
        values_ref = np.nanpercentile(reference.array, percentiles)
        values_self = np.nanpercentile(self.array, percentiles)
        
        out_array = self.array - values_self[0]
        out_array = out_array / (values_self[1] - values_self[0]) * (values_ref[1] - values_ref[0])
        out_array = out_array + values_ref[0]
        
        out_channel = self.copy()
        out_channel.data.data = out_array
        return out_channel
    
    @staticmethod
    def _array_to_nd(array: np.ndarray | Quantity, target_ndim: int) -> np.ndarray | Quantity:
        """Return array with new dimensions filled in from the start if necessary"""
        if array.ndim > target_ndim:
            raise ValueError(f'Got a {array.ndim}-dimensional data array with a target of {target_ndim} dimensions')
        while array.ndim < target_ndim:
            array = array[np.newaxis, :]
        return array
    
    @staticmethod
    def _ensure_increasing_coords(data: xr.DataArray) -> xr.DataArray:
        """Ensure that the axes are all in increasing order, to avoid later problems with slicing order"""
        for coord_name, coord in data.coords.items():
            if len(coord.data) == 1:   # Coordinate is just one point
                continue 
            if coord.data[1] < coord.data[0]:   # Coordinate axis is in decreasing order (assuming monotonic coords)
                data = data.isel({coord_name: slice(None, None, -1)})   # Reverse coordinates to get increasing order
        return data
