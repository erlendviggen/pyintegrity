"""
Module for W2/W1 processing of pulse-echo waveforms. This processing algorithm was first proposed by Havira (1981) and
subsequently refined by later researchers. This module mainly follows the specification of Kimball (1992), but also uses
the conversion between W2/W1 values and impedance specified by Catala et al. (1987).

Furthermore, we have included an extension where W1 and W2 value may optionally be calculated from the waveform envelope
instead of the rectified waveform as specified in the literature. The additional computational load is marginal, and it
reduces random variations in W2.

Note that the W2/W1 algorithm only provides an estimate of the impedance of the material behind the casing. It does not
provide an estimate of the casing thickness.

References:

* Havira, R.M. (1981): “Method and apparatus for acoustically investigating a casing and cement bond in a borehole”.
  US Patent 4,255,798.
* Catala, G., Stowe, I., Henry, D. (1987): “Method for evaluating the quality of cement surrounding the casing of a
  borehole”. US Patent 4,703,427.
* Kimball, C. V. (1992): “Improved processing for oil well cement evaluation—A study with theoretical and laboratory
  data”. In IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control vol. 39, no.1.
"""
import logging
import numpy as np
from scipy.signal import hilbert
from numba import guvectorize, float32, int64
from pyintegrity import Quantity
from ..series import PulseEchoSeries
from ...logchannel import LogChannel
from .result import ProcessingResult
from .helpers import find_peak


class W2W1Result(ProcessingResult):
    """Result container for W2/W1 processing
    
    Args:
        w2w1: Raw W2/W1 values from processing, with no calibration applied

    Attributes:
        w2w1 (LogChannel): Raw W2/W1 for every waveform
        w2w1_fp (Quantity | None): Free-pipe value of W2/W1
        w2w1_cal (LogChannel | None): Raw W2/W1 normalized by free-pipe value of W2/W1
        impedance_fp (Quantity | None): Specified free-pipe impedance
    """
    def __init__(self, w2w1: LogChannel) -> None:
        super().__init__()                          # Initializes impedance behind casing and casing thickness
        self.w2w1 = w2w1                            # Raw W2/W1 for every waveform
        self.w2w1_fp: Quantity | None = None        # Free-pipe value of W2/W1
        self.w2w1_cal: LogChannel | None = None     # Raw W2/W1 divided by free-pipe value of W2/W1
        self.impedance_fp: Quantity | None = None   # Specified free-pipe impedance

    def calibrate_w2w1(self,
                       w2w1_fp: Quantity | None = None,
                       interval_fp: tuple[float, float] | None = None
                       ) -> None:
        """Calibrate W2/W1 values by a free-pipe value which is provided or calculated from a provided depth interval

        Args:
            w2w1_fp: Provided free-pipe W2/W1 value
            interval_fp: Free pipe interval to calculate W2/W1 value from if w2w1_fp is not specified
        """
        if w2w1_fp is None:
            if interval_fp is None:
                raise ValueError('Either w2w1_fp or interval_fp must be not-None')
            # Take the median W2/W1 value inside the free-pipe interval. (Median is robust to casing collar outliers.)
            w2w1_fp = self.w2w1.data.sel(z=slice(interval_fp[0], interval_fp[1])).median().data
        
        self.w2w1_fp = w2w1_fp
        w2w1_cal = self.w2w1.copy()
        w2w1_cal.data = self.w2w1.data / w2w1_fp
        self.w2w1_cal = w2w1_cal

    def calculate_impedance(self,
                            series: PulseEchoSeries,
                            impedance_fp: Quantity,
                            factor_a: Quantity | None = None,
                            factor_b: Quantity | None = None) -> None:
        """Converts normalized W2/W1 values into impedance values

        Args:
            series: `PulseEchoSeries` object containing the waveform data
            impedance_fp: Assumed free-pipe impedance value
            factor_a: Factor for casing thickness and logarithmic W2/W1; corresponds to A in Catala et al. (1987)
            factor_b: Factor for logarithmic W2/W1; corresponds to B in Catala et al. (1987)
        """
        assert self.w2w1_cal is not None
        assert series.casing is not None and series.casing.thickness_nominal is not None
        thickness = series.casing.thickness_nominal
        
        self.impedance_fp = impedance_fp
        impedance = w2w1_to_impedance(self.w2w1_cal, thickness, impedance_fp, factor_a, factor_b)
        self.impedance = LogChannel(impedance, z=self.w2w1.z, phi=self.w2w1.phi)
        

def process_w2w1(series: PulseEchoSeries,
                 use_envelope: bool = False,
                 w2w1_fp: Quantity | None = None,
                 interval_fp: tuple[float, float] | None = None,
                 impedance_fp: Quantity | None = None,
                 factor_a: Quantity | None = None,
                 factor_b: Quantity | None = None
                 ) -> W2W1Result:
    """Perform W2/W1 on input waveform data

    Args:
        series: `PulseEchoSeries` containing input waveform data
        use_envelope: Whether to perform W2/W1 processing on the waveform envelope or the rectified waveform
        w2w1_fp: Optional free-pipe W2/W1 value to use for normalization
        interval_fp: Optional free-pipe interval to calculate `w2w1_fp` from if it is not specified
        impedance_fp: The reference impedance value in the free-pipe interval
        factor_a: Factor for casing thickness and logarithmic W2/W1; corresponds to A in Catala et al. (1987)
        factor_b: Factor for logarithmic W2/W1; corresponds to B in Catala et al. (1987)

    Returns:
        A `W2W1Result` object with the processing results
    """
    waveforms = series.data.to_numpy()   # depth-angle-time array of waveforms
    if use_envelope:
        signals = np.abs(hilbert(waveforms, axis=2))
    else:
        signals = np.abs(waveforms)
    
    w1s, k_w1s = _get_w1(signals, waveforms)
    last_zero_crossings = _previous_zero_crossing(waveforms, k_w1s, axis=2)   #pylint: disable=typecheck
    w2s = _get_w2(signals, last_zero_crossings, series)
    
    w2w1 = LogChannel(w2s/w1s, z=series.z, phi=series.phi)
    result = W2W1Result(w2w1)
    
    # Normalize W2/W1 against its free-pipe value
    if w2w1_fp is None and interval_fp is None:   # Not possible to calculate free-pipe W2/W1 value
        return result   # W2/W1 calculation cannot proceed any further
    result.calibrate_w2w1(w2w1_fp=w2w1_fp, interval_fp=interval_fp)
    
    # Calculate impedance from normalized W2/W1 values
    if impedance_fp is None:   # No specified free-pipe impedance
        return result   # W2/W1 calculation cannot proceed any further
    result.calculate_impedance(series, impedance_fp, factor_a, factor_b)
    
    return result


def _get_w1(signals: np.ndarray, waveforms: np.ndarray, fast: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Get the values and indices of the maximum values of the rectified waveforms
    
    Args:
        signals: Signal to calculate peak value from. May be rectified waveforms or waveform envelope.
        waveforms: Unrectified waveforms to calculate waveform peak index from
        fast: Perform faster peak-finding with lower accuracy
    """
    waveforms = np.abs(waveforms)   # Operate on rectified waveforms
    k_w1s = np.argmax(waveforms, axis=2).astype(np.int64)
    if fast:   # Simply find the maximum value of the raw signal; underestimates W1 by a small but random amount
        w1s = np.max(signals, axis=2)
    else:   # Accurately find the peak signal value by quadratic polynomial fitting
        _, w1s = find_peak(np.arange(signals.shape[-1]), signals)
    return w1s, k_w1s


@guvectorize([(float32[:], int64, float32[:])], '(n),()->()', nopython=True, target='parallel')
def _previous_zero_crossing(waveform: np.ndarray, i_current: int, i_crossing: np.ndarray) -> None:
    """Vectorized numpy ufunc to get the decimal index value of the last zero crossing before a given sample

    Args:
        waveform: Waveform to operate on
        i_current: Sample to search for zero-crossings before
        i_crossing: Output variable to write the index of the last zero crossing to
    """
    waveform = waveform[:i_current+1]   # Only consider the waveform until the given sample
    crossings = np.diff(np.sign(waveform))  # ±2 when passing 0 to next sample, ±1 when zero on this or next sample
    
    i_marked = np.where(crossings != 0)[0]
    if len(i_marked) == 0:   # No zero crossing before i_current
        i_crossing[0] = -1   # Return an invalid value to mark no zero crossings. TODO: Better flagging, use enums?
        return
    
    i_last_marked = i_marked[-1]
    if waveform[i_last_marked] == 0:   # The last zero crossing is exactly here
        i_crossing[0] = i_last_marked
        return
    # Zero crossing somewhere between i_last_marked and i_last_marked + 1; find it by linear interpolation
    i_crossing[0] = i_last_marked + waveform[i_last_marked] / (waveform[i_last_marked] - waveform[i_last_marked + 1])


def _get_w2(waveforms: np.ndarray, last_zero_crossings: np.ndarray, series: PulseEchoSeries) -> np.ndarray:
    """Get the summed absolute signal inside the W2 window

    Args:
        waveforms: Waveforms to calculate W2 from
        last_zero_crossings: Sample index of last zero crossing after peak
        sampling_period: Sampling period of waveform

    Returns:
        Depth-angle array of W2 values for all waveforms
    """
    if series.casing is not None and series.casing.outer_diameter_nominal is not None \
            and series.casing.outer_diameter_nominal.to('inch') < Quantity(6+5/8, 'inch'):
        logging.warning('W2 window delay and length not implemented specifically for casing OD smaller than 6 5/8 in')
    
    # Determine the W2 window
    delay = Quantity(19.2, 'µs')
    window_length = Quantity(19.2, 'µs')
    delay_samples = (delay / series.sampling_period).to('').magnitude
    window_length_samples = round(float((window_length / series.sampling_period).to('')))
    i_starts = np.round(last_zero_crossings + delay_samples).astype('int64')
    i_ends = i_starts + window_length_samples

    @guvectorize([(float32[:], int64, int64, float32[:])], '(n),(),()->()', nopython=True, target='parallel')
    def calculate_w2(waveform, i_start, i_end, w2):
        w2[0] = np.sum(np.abs(waveform[i_start:i_end]))
    
    return calculate_w2(waveforms, i_starts, i_ends, axis=2)   #pylint: disable=typecheck


def w2w1_to_impedance(w2w1_cal: LogChannel | np.ndarray,
                      casing_thickness: Quantity,
                      impedance_fp: Quantity = Quantity(1.5, 'MRayl'),
                      factor_a: Quantity | None = None,
                      factor_b: Quantity | None = None) -> Quantity:
    """Convert from normalized W2/W1 values to impedance following Catala et al. (1987)
    
    Args:
        w2w1_cal: (W2/W1) / (W2/W1)_fp values, (W2/W1)_fp being the assumed free-pipe value of W2/W1
        casing_thickness: The thickness of the logged casing
        impedance_fp: The assumed impedance beyond the casing in a free-pipe section
        factor_a: Factor for casing thickness and logarithmic W2/W1; corresponds to A in Catala et al. (1987)
        factor_b: Factor for logarithmic W2/W1; corresponds to B in Catala et al. (1987)
    """
    if factor_a is None:
        factor_a = Quantity(0.3, 'kg/mm^3/s')   # Default value from Catala et al. (1987)
    if factor_b is None:
        factor_b = Quantity(0.2, 'kg/mm^2/s')   # Default value from Catala et al. (1987)

    if isinstance(w2w1_cal, LogChannel):
        w2w1_cal = w2w1_cal.to_numpy()
    factor_c = (factor_a * casing_thickness.to('mm') + factor_b).to('MRayl')
    return impedance_fp - factor_c * np.log(w2w1_cal)
