from typing import Self
import numpy as np
from pyintegrity import Quantity, xr
from ..material import Material
from ..casing import Casing
from ..logchannel import LogChannel
from ..helpers import to_quantity


class PulseEchoSeries(LogChannel):
    """Contains ultrasonic pulse-echo waveform data and supplementary information about the inner fluid and casing

    `PulseEchoSeries` objects are intended as the input for the implemented pulse-echo processing algorithms.

    Args:
        waveforms: 1D-3D array of ultrasonic pulse-echo waveforms, with dimensions in depth-angle-time order
        sampling_freq: Sampling frequency. Must be provided in seconds or as a `Quantity` object with time units.
        z: Depth axis of the waveform array
        phi: Angle axis of the waveform array
        t_0: Time of the first sample for every waveform in the waveform array

    Attributes:
        sampling_freq (Quantity): Waveform sampling frequency
        casing (Casing | None): Casing information
        inner_material (Material | None): Material on the inner fluid
    """
    def __init__(self,
                 waveforms: np.ndarray | Quantity,
                 sampling_freq: float | Quantity,
                 z: np.ndarray | Quantity,
                 phi: np.ndarray | Quantity | None = None,
                 t_0: np.ndarray | Quantity | None = None) -> None:
        self.sampling_freq = to_quantity(sampling_freq, 'Hz')
        z, phi, t = self._conform_axes(waveforms, z, phi)
        
        # Store data
        super().__init__(data=waveforms, z=z, phi=phi, t=t)
        
        # Store waveform time axes
        self.t_abs: xr.DataArray | None
        if t_0 is not None:
            t_0 = LogChannel._array_to_nd(t_0, 2)
            if not t_0.shape[0] == waveforms.shape[0] and t_0.shape[1] == waveforms.shape[1]:
                raise ValueError('PESeries constructor got arrays for waveforms and start times that do not match')
            t_0 = to_quantity(t_0, 's')
            t_abs = t_0[:, :, np.newaxis] + t[np.newaxis, np.newaxis, :]
            self.t_abs = LogChannel(t_abs, z=z, phi=phi, t=t).data
        else:
            self.t_abs = None

        self.casing: Casing | None = None
        self.inner_material: Material | None = None

    @property
    def sampling_period(self):
        """Returns sampling period as the inverse of the sampling frequency"""
        return (1 / self.sampling_freq).to('s')
    
    def get_depth_section(self, top: float, bottom: float) -> Self:
        """Return a copy which only contains data in the specified interval of the depth axis

        Args:
            top: Start of depth interval, in m
            bottom: End of depth interval, in m

        Returns:
            A copy of this `PulseEchoSeries` containing only an interval of the original depth axis
        """
        i_top = list(self.data.z.values).index(self.data.sel(z=top, method='bfill').z)
        i_bot = list(self.data.z.values).index(self.data.sel(z=bottom, method='ffill').z)

        series = self.copy()
        series.data = series.data.isel(z=slice(i_top, i_bot+1))
        if series.inner_material is not None:
            series.inner_material = series.inner_material.get_depth_section(i_top, i_bot+1)

        return series


    def _conform_axes(self, data, depths, angles) -> tuple[Quantity, Quantity, Quantity]:
        """Returns conformed depth, angle, and time axes as `Quantity` objects with the desired units"""
        depths = to_quantity(depths, 'm')
        if angles is None:
            angles = 360 * np.linspace(0, 1, data.shape[1], endpoint=False)
        angles = to_quantity(angles, 'deg')
        sampling_period = self.sampling_period
        times = Quantity(self.sampling_period * np.arange(data.shape[2]), sampling_period.units)
        return depths, angles, times

    def _get_absolute_time_axes(self, waveforms, start_times) -> xr.DataArray | None:
        if start_times is not None and self.t is not None:
            start_times = LogChannel._array_to_nd(start_times, 2)
            if not start_times.shape[0] == waveforms.shape[0] and start_times.shape[1] == waveforms.shape[1]:
                raise ValueError('PESeries constructor got arrays for waveforms and start times that do not match')
            start_times = to_quantity(start_times, 's')
            t_abs = start_times[:, :, np.newaxis] + self.t[np.newaxis, np.newaxis, :]
            t_abs = LogChannel(t_abs, z=self.z, phi=self.phi, t=self.t).data
        else:
            t_abs = None
        return t_abs
