import pint
from pint import RedefinitionError
import pint_xarray
from pint_xarray import unit_registry as ureg
import xarray as xr

# Set up xarray's unit registry to include additional units that we need to parse DLIS files
try:
    ureg.define('[impedance] = [density] * [speed]')
    ureg.define('rayl = (kilogram / meter**3) * (meter / second) = Rayl')
    ureg.define('m3 = meter**3')
    ureg.define('cm3 = centimeter**3')
    ureg.define('@alias lb = lbm')
    ureg.define('s2 = second ** 2')
except RedefinitionError:
    pass

# Get Quantity and Unit classes with the added units available
Quantity = pint.Quantity
Unit = pint.Unit
