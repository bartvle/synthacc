"""
The 'source.faults' module.
"""


import matplotlib.pyplot as plt

from ..apy import Object, is_pos_number
from ..earth.flat import Rectangle
from .moment import calculate as calculate_moment, m0_to_mw


## Average rigidity (in Pa) in crust (from USGS website)
RIGIDITY = 3.2 * 10**10


class SingularFault(Rectangle):
    """
    A fault with a rectangular surface.
    """

    def __init__(self, x1, y1, x2, y2, upper_depth, lower_depth, dip, rigidity=RIGIDITY, validate=True):
        """
        """
        super().__init__(
            x1, y1, x2, y2, upper_depth, lower_depth, dip, validate=validate)

        if validate is True:
            assert(is_pos_number(rigidity))

        self._rigidity = rigidity

    @property
    def rigidity(self):
        """
        return: pos number
        """
        return self._rigidity

    def get_max_moment(self, slip, validate=True):
        """
        slip: pos number, slip (in m)

        return: pos number, maximum seismic moment
        """
        return calculate_moment(self.area, slip, self.rigidity, validate)

    def get_max_magnitude(self, slip, validate=True):
        """
        slip: pos number, slip (in m)

        return: number, maximum moment magnitude
        """
        return m0_to_mw(self.get_max_moment(slip, validate), validate=False)


class ComposedFault(Object):
    """
    A fault composed of multiple singular faults.
    """

    def __init__(self, faults, validate=True):
        """
        faults: list of 'faults.SingularFault' instances
        """
        if validate is True:
            for f in faults:
                assert(type(f) is SingularFault)

        self._faults = faults

    def __len__(self):
        """
        """
        return len(self._faults)

    def __getitem__(self, i):
        """
        """
        assert(type(i) is int)
        return self._faults[i]

    def __iter__(self):
        """
        """
        for f in self._faults:
            yield f

    @property
    def length(self):
        """
        return: pos number
        """
        l = 0
        for f in self:
            l += f.length
        return l

    @property
    def area(self):
        """
        return: pos number
        """
        a = 0
        for f in self:
            a += f.area
        return a

    def get_max_moment(self, slip, validate=True):
        """
        slip: pos number, slip (in m)

        return: pos number, maximum seismic moment
        """
        if validate is True:
            assert(is_pos_number(slip))

        return sum([f.get_max_moment(slip, validate=False) for f in self])

    def get_max_magnitude(self, slip, validate=True):
        """
        slip: pos number, slip (in m)

        return: number, maximum moment magnitude
        """
        return m0_to_mw(self.get_max_moment(slip, validate), validate=False)
