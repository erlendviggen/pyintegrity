from typing import NewType, cast
import logging
import numpy as np
from dlisio import dlis
from pint import UndefinedUnitError
from pyintegrity import Quantity
from ..helpers import to_quantity
from ..material import Material
from ..casing import Casing
from ..logchannel import LogChannel
from ..logparameter import LogParameter
from .series import PulseEchoSeries


# Define new type to improve type readability
FrameName = NewType('FrameName', str)
ChannelName = NewType('ChannelName', str)
ParameterName = NewType('ParameterName', str)


def get_usit_data(file: dlis.LogicalFile) -> tuple[dict[FrameName, dict[ChannelName, LogChannel]],
                                                   dict[ParameterName, LogParameter]]:
    """Get all USIT-specific channels and parameters from a DLIS logical file

    Args:
        file: The DLIS logical file to extract data from

    Returns:
        USIT channels as `LogChannel` objects, organized by frame in a dict, and USIT parameters in a dict
    """
    usit_frame_channels = _get_usit_channels_by_frame(file)
    usit_frame_curves = _get_usit_curves_by_frame(file, usit_frame_channels)
    usit_frame_logchannels = _to_logchannels(file, usit_frame_channels, usit_frame_curves)
    usit_parameters = _get_dlis_parameters(file, from_tool='.*USIT.*')
    
    return usit_frame_logchannels, usit_parameters


def get_usit_series(file: dlis.LogicalFile) -> PulseEchoSeries:
    """Assemble USIT waveform data into a `PulseEchoSeries` object

    Args:
        file: The DLIS logical file to extract data from

    Returns:
        A `PulseEchoSeries` object containing data about the waveforms and the assumed material and casing parameters
    """
    usit_frame_channels = _get_usit_channels_by_frame(file)
    usit_frame_curves = _get_usit_curves_by_frame(file, usit_frame_channels)

    waveform_curve, waveform_t_0_curve, waveform_index_curve = _get_usit_waveforms(file, usit_frame_curves)

    series = PulseEchoSeries(waveform_curve, sampling_freq=Quantity(2, 'MHz'),
                             z=waveform_index_curve, t_0=waveform_t_0_curve)
    series.casing = _get_casing(file)
    series.inner_material = _get_inner_material(file, usit_frame_curves, waveform_index_curve)

    return series


def _get_usit_channels_by_frame(file: dlis.LogicalFile) -> dict[FrameName, list[dlis.Channel]]:
    """Get all channels belonging to the USIT tool, organized by frame"""
    [usit] = file.find('TOOL', '.*USIT.*')
    usit_channels: list[dlis.Channel] = [ch for ch in usit.channels if ch.frame is not None]
    usit_frame_channels: dict[FrameName, list[dlis.Channel]] = {}
    
    for channel in sorted(usit_channels, key=lambda channel: channel.name):
        usit_frame_channels.setdefault(channel.frame.name, []).append(channel)
    
    return usit_frame_channels


def _get_usit_curves_by_frame(file: dlis.LogicalFile,
                              usit_frame_channels: dict[FrameName, list[dlis.Channel]]
                              ) -> dict[FrameName, np.ndarray]:
    """Get all USIT curves and their index curves, all organized by frame"""
    usit_frames: list[dlis.Frame] = [file.object('FRAME', framename) for framename in usit_frame_channels.keys()]
    usit_frame_curves: dict[FrameName, np.ndarray] = {}
    
    for frame in usit_frames:
        index_channel = next(ch for ch in frame.channels if ch.name == frame.index)
        usit_frame_curves[frame.name] = frame.curves()[[index_channel.name]
                                                        + sorted([ch.name for ch in usit_frame_channels[frame.name]])]
    
    return usit_frame_curves


def _to_logchannels(file: dlis.LogicalFile,
                    usit_frame_channels: dict[FrameName, list[dlis.Channel]],
                    usit_frame_curves: dict[FrameName, np.ndarray]
                    ) -> dict[FrameName, dict[ChannelName, LogChannel]]:
    """Get all USIT channels as LogChannel objects, organized by frame"""
    # Prepare angle and time axes
    n_angles = file.object('PARAMETER', 'NWPD').values[0]
    angles = Quantity(np.linspace(0, 360, n_angles, endpoint=False), 'deg')
    n_samples = file.object('PARAMETER', 'NPPW').values[0]
    times = Quantity(np.arange(n_samples))   # Sample numbers instead of time axis
    
    usit_frame_logchannels: dict[FrameName, dict[ChannelName, LogChannel]] \
        = {framename: {} for framename in usit_frame_channels.keys()}
    for framename in usit_frame_channels.keys():
        channels = usit_frame_channels[framename]
        curves = usit_frame_curves[framename]
        logchannels = usit_frame_logchannels[framename]
        index_curve = _get_index_curve(channels[0].frame, curves)
        
        for channel in channels:
            channel_info = {'name': channel.name, 'description': channel.long_name}
            if channel.units:
                curve = curves[channel.name] * Quantity(channel.units)
            else:
                curve = Quantity(curves[channel.name])
            if curve.ndim == 1:
                logchannels[channel.name] = LogChannel(curve, z=index_curve, **channel_info)
            elif curve.ndim == 2 and curve.shape[1] == len(angles):
                logchannels[channel.name] = LogChannel(curve, z=index_curve, phi=angles, **channel_info)
            elif curve.ndim == 2 and curve.shape[1] == len(times):
                logchannels[channel.name] = LogChannel(curve, z=index_curve, t=times, **channel_info)
            else:
                logging.warning(f'Cannot handle channel {channel.name} with shape {curve.shape}, skipping...')

    return usit_frame_logchannels


def _get_dlis_parameters(file: dlis.LogicalFile,
                         from_tool: str = '') -> dict[ParameterName, LogParameter]:
    """Get parameters; can be filtered by tool"""
    if from_tool:
        [tool] = file.find('TOOL', from_tool)
        raw_parameters = tool.parameters
    else:
        raw_parameters = file.parameters
    
    parameters: dict[ParameterName, LogParameter] = {}
    for parameter in sorted(raw_parameters, key=lambda parameter: parameter.name):
        if parameter.name in parameters:
            logging.warning(f'Overwriting duplicate parameter {parameters[parameter.name]}')
        try:
            parameters[parameter.name] = LogParameter(parameter)
        except UndefinedUnitError:
            units = parameter.attic["VALUES"].units
            logging.warning(f'Skipping parameter {parameter.name} due to unhandled units {units}')

    return parameters


def _get_usit_waveforms(file: dlis.LogicalFile,
                        usit_frame_curves: dict[FrameName, np.ndarray]
                        ) -> tuple[Quantity, Quantity, Quantity]:
    """Get re-equalized waveforms, their start times, and their depth index"""
    frame: dlis.Frame = file.object('CHANNEL', 'U001').frame
    curves = usit_frame_curves[frame.name]

    waveform_curve = _assemble_waveform_curve(file, frame, curves)
    waveform_t_0_curve = _get_t_0_curve(file, frame, curves)
    waveform_index_curve = _get_index_curve(frame, curves)

    return waveform_curve, waveform_t_0_curve, waveform_index_curve


def _assemble_waveform_curve(file: dlis.LogicalFile, frame: dlis.Frame, curves: np.ndarray) -> Quantity:
    """Assemble USIT waveforms from individual depth-time arrays into a depth-angle-time array and re-equalize it"""
    def azim_to_name(j) -> ChannelName:
        """Return the name of the waveform channel corresponding to azimuth index j"""
        return cast(ChannelName, f'U{j + 1:03}')
    
    n_depths = curves['U001'].shape[0]
    n_angles = file.object('PARAMETER', 'NWPD').values[0]
    n_samples = file.object('PARAMETER', 'NPPW').values[0]
    waveform_curve = Quantity(np.zeros((n_depths, n_angles, n_samples), dtype='float32'))
    for j in range(n_angles):
        waveform_curve[:, j, :] = curves[azim_to_name(j)]

    scaling_channel: dlis.Channel = file.object('CHANNEL', 'WAGN')
    if scaling_channel.frame != frame:
        raise RuntimeError('The scaling channel WAGN and the waveform channel U001 have different depth axes')
    waveform_scaling = 10**(-curves['WAGN']/20)
    waveform_curve *= waveform_scaling[:, :, np.newaxis]
    
    return waveform_curve


def _get_t_0_curve(file: dlis.LogicalFile, frame: dlis.Frame, curves: np.ndarray) -> Quantity:
    """Get the starting sample time for each waveform as a depth-angle array"""
    channel_name = 'WFDL'
    t_0_channel: dlis.Channel = file.object('CHANNEL', channel_name)
    if t_0_channel.frame != frame:
        raise RuntimeError('The start time channel WFDL and the waveform channel U001 have different depth axes')
    t_0_curve: Quantity = curves[channel_name] * Quantity(t_0_channel.units).to('s')
    correction = to_quantity(file.object('PARAMETER', 'USTO'), 's')
    return t_0_curve + correction


def _get_index_curve(frame: dlis.Frame, curves: np.ndarray) -> Quantity:
    """Get the depth index curve of a frame"""
    index_channel = next(ch for ch in frame.channels if ch.name == frame.index)
    index_curve = curves[frame.index] * Quantity(index_channel.units).to('m')
    return index_curve


def _get_casing(file: dlis.LogicalFile) -> Casing:
    speed = to_quantity(file.object('PARAMETER', 'VCAS'), 'm/s')
    impedance = to_quantity(file.object('PARAMETER', 'ZCAS'), 'MRayl')
    density = to_quantity(file.object('PARAMETER', 'CSDE'), 'kg/m^3')
    
    derived_density = (impedance / speed).to('kg/m^3')
    if np.abs(derived_density - density) > Quantity(1, 'kg/m^3'):
        logging.warning(f'Ignoring log-specified casing density {density:.1f~P} ' + \
                        f'due to mismatch with derived density {derived_density:.1f~P}')
        material = Material(speed=speed, impedance=impedance)
    else:
        material = Material(speed=speed, impedance=impedance, density=density)
    
    outer_diameter_nominal = to_quantity(file.object('PARAMETER', 'CSIZ'), 'm')
    thickness_nominal = to_quantity(file.object('PARAMETER', 'THNO'), 'm')

    return Casing(material=material, outer_radius_nominal=outer_diameter_nominal/2, thickness_nominal=thickness_nominal)


def _get_inner_material(file: dlis.LogicalFile,
                        usit_frame_curves: dict[FrameName, np.ndarray],
                        waveform_index_curve: Quantity) -> Material:
    def get_material_curve(channel_name: str, target_units: str) -> Quantity:
        channel: dlis.Channel = file.object('CHANNEL', channel_name)
        curve = Quantity(usit_frame_curves[channel.frame.name][channel_name], channel.units)
        curve = to_quantity(curve, target_units)   # Cannot just use .to() when converting slownesses to speeds
        
        if len(curve) != len(waveform_index_curve):   # This channel and waveform channels are in different frames
            # Need to interpolate the curve in depth to match the waveforms
            frame = channel.frame
            index_channel = next(ch for ch in frame.channels if ch.name == frame.index)
            index_curve = usit_frame_curves[frame.name][frame.index] * Quantity(index_channel.units).to('m')
            curve = LogChannel(curve, z=index_curve).interpolate_to(z=waveform_index_curve).array

        return curve
    
    speed_curve = get_material_curve('CFVL', 'm/s')
    impedance_curve = get_material_curve('CZMD', 'MRayl')
    return Material(speed=speed_curve, impedance=impedance_curve)
