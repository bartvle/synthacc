"""
The 'units' module.
"""


from .apy import Object, is_number, is_string


SI_PREFIXES = [
    ('yotta', 'Y', 1e+24),
    ('zetta', 'Z', 1e+21),
    (  'exa', 'E', 1e+18),
    ( 'peta', 'P', 1e+15),
    ( 'tera', 'T', 1e+12),
    ( 'giga', 'G', 1e+09),
    ( 'mega', 'M', 1e+06),
    ( 'kilo', 'k', 1e+03),
    ('hecto', 'h', 1e+02),
    ( 'deca', 'da',1e+01),
    ( 'deci', 'd', 1e-01),
    ('centi', 'c', 1e-02),
    ('milli', 'm', 1e-03),
    ('micro', 'u', 1e-06),
    ( 'nano', 'n', 1e-09),
    ( 'pico', 'p', 1e-12),
    ('femto', 'f', 1e-15),
    ( 'atto', 'a', 1e-18),
    ('zepto', 'z', 1e-21),
    ('yocto', 'y', 1e-24),
]


# SI units for motion
MOTION_SI = {'dis': 'm', 'vel': 'm/s', 'acc': 'm/s2'}


class Unit(Object):
    """
    A unit of a quantity.
    """

    def __init__(self, symbol, si_scale, quantity, validate=True):
        """
        symbol: string
        si_scale: number
        quantity: string
        """
        if validate is True:
            assert(is_string(symbol))
            assert(is_number(si_scale))
            assert(is_string(quantity))

        self._symbol = symbol
        self._si_scale = si_scale
        self._quantity = quantity

    @property
    def symbol(self):
        """
        return: string
        """
        return self._symbol

    @property
    def si_scale(self):
        """
        return: number
        """
        return self._si_scale

    @property
    def quantity(self):
        """
        return: number
        """
        return self._quantity

    def __truediv__(self, unit):
        """
        Divide two units by dividing their SI scale.

        Example: m/cm = 1/0.01 = 100

        return: number
        """
        assert(type(unit) is self.__class__)
        assert(unit.quantity == self.quantity)
        return self.si_scale / unit.si_scale


def make_motion_units():
    """
    Make base and prefixed units for motion.
    """
    bases = (
        Unit('m',    1.,   'displacement'),
        Unit('m/s',  1.,       'velocity'),
        Unit('m/s2', 1.,   'acceleration'),
        Unit('g',    9.81, 'acceleration'),
        )
    units = {}
    for base in bases:
        units[base.symbol] = base
        for (name, symbol, scale) in SI_PREFIXES:
            symbol += base.symbol
            assert(symbol not in units)
            si_scale = base.si_scale * scale
            quantity = base.quantity
            units[symbol] = Unit(symbol, si_scale, quantity)
    return units


## units
MOTION = make_motion_units()
