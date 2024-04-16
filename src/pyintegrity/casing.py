from pyintegrity import Quantity
from .material import Material


class Casing:
    """Contains information on a casing

    Args:
        material: A `Material` object specifying the casing's material
        outer_radius_nominal: A `Quantity` object with length units describing the nominal outer radius
        thickness_nominal: A `Quantity` object with length units describing the nominal casing thickness

    Attributes:
        material (Material | None): The casing material
        outer_radius_nominal (Quantity | None): The outer radius of the casing
        thickness_nominal (Quantity | None): The casing thickness
    """
    def __init__(self,
                 material: Material | None = None,
                 outer_radius_nominal: Quantity | None = None,
                 thickness_nominal: Quantity | None = None) -> None:
        self.material = material
        self.outer_radius_nominal = outer_radius_nominal
        self.thickness_nominal = thickness_nominal

    @property
    def outer_diameter_nominal(self) -> Quantity | None:
        """Returns nominal outer diameter"""
        if self.outer_radius_nominal is None:
            return None
        return 2 * self.outer_radius_nominal
    
    @property
    def inner_radius_nominal(self) -> Quantity | None:
        """Returns nominal inner radius"""
        if self.outer_radius_nominal is None or self.thickness_nominal is None:
            return None
        return self.outer_radius_nominal - self.thickness_nominal
    
    @property
    def inner_diameter_nominal(self) -> Quantity | None:
        """Returns nominal inner diameter"""
        if self.inner_radius_nominal is None:
            return None
        return self.inner_radius_nominal * 2
