# Fundamentals

These are the fundamental classes and functions used by Pyintegrity.


## Basic unit tools

::: pyintegrity.Quantity
Pyintegrity uses the Pint library's `Quantity` object to provide numerical values or arrays with physical units. `pyintegrity.Quantity` should be used instead of `pint.Quantity`, as the former supports a few additional units that are used in ultrasonic well logs.

::: pyintegrity.Unit
Similarly to `pyintegrity.Quantity`, `pyintegrity.Unit` should be used in place of `pint.Unit` if necessary. However, it is usually more convenient to use strings to represent units when building `Quantity` objects.

::: pyintegrity.helpers.to_quantity


## Log information

In a well log file on the DLIS format, the majority of information is provided as log parameters and log channels. Pyintegrity has internal representations for these types of objects.

::: pyintegrity.logparameter.LogParameter
::: pyintegrity.logchannel.LogChannel


## Containers

These container classes are used for well-related information.

::: pyintegrity.material.Material
::: pyintegrity.casing.Casing

