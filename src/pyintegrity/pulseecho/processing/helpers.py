from typing import overload
import numpy as np
from scipy.signal import hilbert
from numba import guvectorize, float32, float64
from pyintegrity import Quantity


@overload
def find_peak(xs: np.ndarray, ys: np.ndarray, use_envelope: bool = False) -> tuple[np.ndarray, np.ndarray]:
    ...
@overload
def find_peak(xs: Quantity, ys: Quantity, use_envelope: bool = False) -> tuple[Quantity, Quantity]:
    ...
@overload
def find_peak(xs: Quantity, ys: np.ndarray, use_envelope: bool = False) -> tuple[Quantity, np.ndarray]:
    ...
@overload
def find_peak(xs: np.ndarray, ys: Quantity, use_envelope: bool = False) -> tuple[np.ndarray, Quantity]:
    ...
def find_peak(xs: np.ndarray | Quantity,
              ys: np.ndarray | Quantity,
              use_envelope: bool = False) -> tuple[np.ndarray | Quantity, np.ndarray | Quantity]:
    """Accurately determine peak coordinate and height of signals using quadratic polynomial fit

    `xs` and `ys` may be either 1D arrays representing a single signal, or ND arrays representing multiple signals. In
    the latter case, the last dimension in `xs` and `ys` is assumed to represent signal samples, and the returned arrays
    will retain the shape of the other dimensions of `ys`. The combination of an 1D coordinate array `xs` and an ND
    signal value array `ys` is also possible, given that the last dimension of `ys` has the same number of elements as
    `xs`.

    Args:
        xs: Coordinate array; may be a `Quantity` array or a numpy array
        ys: Signal value array; may be a `Quantity` array or a numpy array
        use_envelope: Whether to find the peak of the signal envelope instead of using the signal `y` directly

    Returns:
        A tuple containing the peak x and y coordinates as arrays or `Quantity` objects, depending on the input types
    """
    if not xs.shape[-1] == ys.shape[-1]:
        raise RuntimeError('find_peak() got x and y inputs of incompatible size')

    # Convert xs and ys to numpy arrays, but keep any information on units
    xs, x_units = (xs.magnitude, xs.units) if isinstance(xs, Quantity) else (xs, None)
    ys, y_units = (ys.magnitude, ys.units) if isinstance(ys, Quantity) else (ys, None)

    if use_envelope:   # Find peak of signal envelope instead of raw signal
        ys = np.abs(hilbert(ys, axis=-1))
    
    x_peak: np.ndarray | Quantity
    y_peak: np.ndarray | Quantity
    x_peak, y_peak = _find_peak_vectorized(xs, ys, axis=-1)   #pylint: disable=typecheck,unpacking-non-sequence

    # Reapply units, where appropriate
    if x_units is not None:
        x_peak = Quantity(x_peak, x_units)
    if y_units is not None:
        y_peak = Quantity(y_peak, y_units)

    return x_peak, y_peak


@guvectorize([(float32[:], float32[:], float32[:], float32[:]),
              (float64[:], float64[:], float64[:], float64[:])], '(n),(n)->(),()', nopython=True, target='parallel')
def _find_peak_vectorized(x: np.ndarray, y: np.ndarray, x_peak: np.ndarray, y_peak: np.ndarray):
    """Vectorized numpy ufunc to accurately estimate the peak value of one or more signals by quadratic polynomial fit

    Args:
        x: 1D coordinate array for a single signal
        y: 1D signal value array for a single signal
        x_peak: Output variable for the signal peak coordinate
        y_peak: Output variable for the signal peak value
    """
    i_max = np.argmax(y)
    i_range = i_max + np.array([-1, 0, 1])
    if i_range[0] < 0 or i_range[2] >= len(x):
        # The peak is on the edge; give up on this signal
        x_peak[0] = np.nan
        y_peak[0] = np.nan
        return
    x, y = x[i_range], y[i_range]
        
    # Can't use np.polyfit in Numba; using direct quadratic solution from https://stackoverflow.com/a/717791/3468067
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    a = x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])
    b = x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0]) + x[0]**2 * (y[1] - y[2])
    c = x[1] * x[2] * (x[1] - x[2]) * y[0] + x[2] * x[0] * (x[2] - x[0]) * y[1] + x[0] * x[1] * (x[0] - x[1]) * y[2]
    x_peak[0] = - b / (2*a)
    y_peak[0] = (c - b**2 / (4*a)) / denom
