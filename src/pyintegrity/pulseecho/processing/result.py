from ...logchannel import LogChannel


class ProcessingResult:
    """Generic result container to be subclassed by containers for specific algorithms
    
    Args:
        impedance: Impedance behind casing estimated by a processing algorithm
        thickness: Casing thickness estimated by a processing algorithm
    """
    def __init__(self,
                 impedance: LogChannel | None = None,
                 thickness: LogChannel | None = None) -> None:
        self.impedance = impedance
        self.thickness = thickness
