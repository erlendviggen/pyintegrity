# pylint: disable=invalid-name
from typing import Any
import json
import pandas as pd
import numpy as np
from pyintegrity import Quantity
 
from ..material import Material
from ..casing import Casing
from ..helpers import to_quantity
from ..pulseecho.series import PulseEchoSeries
from ..model import Model, Dims, Transducer, Gap, Eccentering
from ..logchannel import LogChannel


Jdatatype = dict[str, dict[str, Any]]   # A two-level nested dictionary with strings as keys


def get_comsol_data(filepath: str,filesIDs: int | list):
    """ read json files containing COMSOL model result creating a pandas dataframe containing a `PulseEchoSeries` and \
          model parameters related to the id of the file
    
    Args:
        filepath: string to file path
        filesIDs: either an integer, a range or a list to define the files that should be read, if a list with two 
            values is given, this is interpreted as a range
        
    Returns:
        (DataFrame): pandas dataframe containing file name in the data base, PulseEchoSeries containing the data, \
            parameter class containing additional metedata from the model
    """
    # Check if filesID is a range, a list or, a file
    if isinstance(filesIDs, int):
        fid=np.zeros([1],dtype=int)
        fid[0]=filesIDs
    elif len(filesIDs)==2:
        fid=np.arange(filesIDs[0],filesIDs[1]+1)
    else:
        fid=np.array(filesIDs)
    # headers for dataframe
    headers=['fid','series','parameters'] 
    fid_series_meta=[]
    # read the differetn files and append the data frame
    for f in fid:
        Jdata = _read_Database(filepath,f) 
        series = _to_PEseries(Jdata)
        metadata = _get_metadata(Jdata)
        fid_series_meta.append(pd.DataFrame([[f,series,metadata]], columns=headers))
    return pd.concat(fid_series_meta, ignore_index=True)


def _get_metadata(Jdata: Jdatatype) -> Model:
    """combine meta data from the model
    
    Args:
        Jdata: nested dictionary containing all the information from the json file
    """
    
    out=Material(speed=Jdata['outside']['vp'],density=Jdata['outside']['rho'])
    gapM=Material(speed=Jdata['gap']['vp'],density=Jdata['gap']['rho'])
    gap=Gap(material=gapM,thickness=Jdata['gap']['thickness'])
    ecc=Eccentering(incident_angle=Jdata['eccentering']['incident_angle'],
                    transducer_pipe_distance=Jdata['eccentering']['transducer_pipe_distance'])
    transd=Transducer(diameter=Jdata['transducer']['diameter'],
                      focal_length=Jdata['transducer']['focal_length'],
                      apodization=Jdata['transducer']['apodization'], 
                      pulse_bw=Jdata['pulse']['bw'],
                      pulse_max_t0=Jdata['pulse']['tp0'],
                      pulse_f0=Jdata['pulse']['f0'],
                      transducer_pipe_distance=Jdata['inside']['transducer_pipe_distance'])
    dim=Dims(domain=Jdata['general']['domain'],
             geometry=Jdata['general']['geometry'],
             comsol_interface=Jdata['general']['interface'],
             symmetries=Jdata['general']['symmetries'])
    
    return Model(dimensions=dim,transducer=transd,gap=gap,eccentering=ecc, outside=out)

def _read_Database(path2file: str, fid: int) -> Jdatatype:
    """ Read json file
    
    Args:
        path2file: string of path to file location
        fid: file name to read as int
        
    Result:
        Jdata: nested dictionary containing all the data in the json file
    """
    with open(path2file+str(fid)+'.json', encoding='utf-8') as f:
        Jdata = json.load(f)
    
    return Jdata

def _to_waveform(Jdata: Jdatatype) -> tuple[Quantity, Quantity]:
    """read actual waveform data, and cut trace to not include the send pulse
    
    Args:
        Jdata: nested dictionary containing all the information from the json file
    
    Result:
        waveform_curve: vector containing the waveform
        waveform_t_0_curve: t0 start point
    """
    tvec=np.arange(Jdata['waveform']['t0'],
                   Jdata['waveform']['sample_no']*Jdata['waveform']['dt'],
                   Jdata['waveform']['dt'])
    data=np.array(Jdata['waveform']['values'])
    # cut away send pulse 
    tcut=Jdata['pulse']['tp0']*2
    mask = tvec>=tcut
    tvec=tvec[mask]
    data=data[mask]
    waveform_curve = Quantity(np.zeros((1, 1, len(data)), dtype='float64'))
    waveform_curve[0,0,:] = data
    t_0=np.zeros([1,1])
    t_0[0,0]=tvec[0]
    waveform_t_0_curve =Quantity(t_0,'s')

    return waveform_curve, waveform_t_0_curve

def _get_casing_model(Jdata: Jdatatype) -> Casing:
    """write Casing class data
    
    Args:
        Jdata: nested dictionary containing all the information from the json file
    
    Result:
        Casing: casing metadata
    """
    material = Material(speed=Jdata['interface']['vp'], density=Jdata['interface']['rho'])
    thickness_nominal = to_quantity(Jdata['interface']['thickness'], 'm')
    if Jdata['interface']['outer_diameter'] is None:
        return Casing(material=material, thickness_nominal=thickness_nominal)
    else:
        outer_diameter_nominal = to_quantity(Jdata['interface']['outer_diameter'], 'm')

    return Casing(material=material, outer_radius_nominal=outer_diameter_nominal/2, thickness_nominal=thickness_nominal)

def _get_inner_material(Jdata: Jdatatype) -> Material:
    """write material parameters of inner material
    
    Args:
        Jdata: nested dictionary containing all the information from the json file
    
    Result:
        Material: material metadata
    """    
    speed=to_quantity(Jdata['inside']['vp'],'m/s')
    density=to_quantity(Jdata['inside']['rho'],'kg/m3')
    impedance=to_quantity(speed*density,'mrayl')
    speed_curve=LogChannel(speed,z=np.array([0]))
    impedance_curve=LogChannel(impedance,z=np.array([0]))
    density_curve=LogChannel(density,z=np.array([0]))
    return Material(speed=speed_curve,impedance=impedance_curve, density=density_curve)
    
    
def _to_PEseries(Jdata: Jdatatype) -> PulseEchoSeries:
    """write PulseEcho Series from json file data
    
    Args:
        Jdata: nested dictionary containing all the information from the json file
        
    Result:
        series: PulseEchoSeries containing waveform and metadata
    """
    waveform_curve, waveform_t_0_curve =_to_waveform(Jdata)
    series = PulseEchoSeries(waveform_curve, sampling_freq=to_quantity(Jdata['waveform']['dt'],'s'),
                             z=np.array([0]), phi=np.array([0]), t_0=waveform_t_0_curve[0])
    series.casing =_get_casing_model(Jdata)
    series.inner_material = _get_inner_material(Jdata)
    # series.inner_material = Material(speed=Jdata['inside']['vp'], density=Jdata['inside']['rho'])
    return series
