from pyintegrity import Quantity
from .helpers import to_quantity
from .material import Material


class Dims:
    """ Class to collect all the dimension information related to the Comsol model domain
    
    Args:
        domain: 2.5D or 3D
        geometry: plate or pipe
        comsol_interface: time explicit or time domain
        symmetries: axisymmetric, 2 symmetries, or 1 symmetry
    """
    def __init__(self, 
                 domain: float | None = None,
                 geometry: str | None = None,
                 comsol_interface: str | None = None,
                 symmetries: str | None = None) -> None:
        self.domain=domain
        self.geometry=geometry
        self.comsol_interface=comsol_interface
        self.symmetries=symmetries

class Transducer:
    """ Class to collect all the transducer information related to the Comsol model domain
    
    Args:
        diameter: transducer diameter size
        focal_length: transducer focal length
        apodization: apodization when sending the pulse
        pulse_bw: bandwidth of the pulse
        pulse_max_t0: time of maximum of the pulse
        pulse_f0: pulse frequency
        transducer_pipe_distance: distance between transducer center and pipe
    """
    def __init__(self, 
                 diameter: float | Quantity | None = None,
                 focal_length: float | Quantity | None = None,
                 apodization: str | None = None,
                 pulse_bw: float | Quantity | None = None,
                 pulse_max_t0: float | Quantity | None = None,
                 pulse_f0: float | Quantity | None = None,
                 transducer_pipe_distance: float | Quantity | None = None) -> None:
        
        if diameter is not None:
            diameter = to_quantity(diameter, 'm')
        self.diameter=diameter
        
        if focal_length is not None:
            focal_length = to_quantity(focal_length, 'm')
        self.focal_length=focal_length
        
        self.apodization=apodization
        
        self.pulse_bw=pulse_bw
        
        if pulse_max_t0 is not None:
            pulse_max_t0 = to_quantity(pulse_max_t0, 's')
        self.pulse_max_t0=pulse_max_t0
        
        if pulse_f0 is not None:
            pulse_f0 = to_quantity(pulse_f0, 'Hz')
        self.pulse_f0=pulse_f0
        
        if transducer_pipe_distance is not None:
            transducer_pipe_distance = to_quantity(transducer_pipe_distance, 'm')
        self.transducer_pipe_distance=transducer_pipe_distance
       

class Gap:
    """ Class to collect all the information if a gap exists between the pipe
    and the outside material in the Comsol model 
    
    Args:
        material: Material class containing the parameters of the material in the gap
        thickness: thickness of the gap
    """
    def __init__(self,
                 material: Material | None = None,
                 thickness:  float | Quantity | None = None) -> None:
        
        self.material=material
        
        if thickness is not None:
            thickness = to_quantity(thickness, 'm')
        self.thickness=thickness
        
class Eccentering:
    """ Class to collect all the information if a gap exists between the pipe
    and the outside material in the Comsol model 
    
    Args:
        incident_angle: incident angle on the pipe
        transducer_pipe_distance: actual transducer pipe distance in case of eccentering
    """
    def __init__(self,
                 incident_angle: float | Quantity | None,
                 transducer_pipe_distance: float | Quantity | None) -> None:
        
        if incident_angle is not None:
            incident_angle = to_quantity(incident_angle, 'deg')
        self.incident_angle=incident_angle
        
        if transducer_pipe_distance is not None:
            transducer_pipe_distance = to_quantity(transducer_pipe_distance, 'm')
        self.transducer_pipe_distance=transducer_pipe_distance
        


class Model():
    """ Class to collect all the metadata from the subclasses
    
    Args:
        dimensions: Dims class
        transducer: Transducer class
        gap: Gap class
        eccentering: Eccentering class
        outside: outside class
    """
    def __init__(self, 
                dimensions: Dims | None = None,
                transducer: Transducer | None = None,   # transducer, pulse, transducer pipe distance
                gap: Gap | None = None,
                eccentering: Eccentering | None = None,
                outside: Material | None = None)-> None:
        self.dimensions=dimensions
        self.transducer=transducer
        self.gap=gap
        self.eccentering=eccentering
        self.outside=outside
