"""
The 'source.faults' module.
"""


import random

from ..apy import Object, is_non_neg_number, is_pos_number
from ..earth import flat as earth


## Average rigidity (in Pa) in crust (from USGS website)
RIGIDITY = 3.2 * 10**10


class SimpleFault(earth.SimpleSurface):
    """
    A fault with a rectangular surface.
    """

    def __init__(self, x1, y1, x2, y2, upper_depth, lower_depth, dip, rigidity=RIGIDITY, upper_sd=None, lower_sd=None, validate=True):
        """
        """
        super().__init__(
            x1, y1, x2, y2, upper_depth, lower_depth, dip, validate=validate)

        if upper_sd is None:
            upper_sd = upper_depth
        if lower_sd is None:
            lower_sd = lower_depth

        if validate is True:
            assert(is_pos_number(rigidity))
            assert(is_non_neg_number(upper_sd) and upper_sd >= upper_depth)
            assert(is_non_neg_number(lower_sd) and lower_sd <= lower_depth)

        self._rigidity = rigidity
        self._upper_sd = upper_sd
        self._lower_sd = lower_sd

    @property
    def rigidity(self):
        """
        return: pos number
        """
        return self._rigidity

    @property
    def upper_sd(self):
        """
        """
        return self._upper_sd

    @property
    def lower_sd(self):
        """
        """
        return self._lower_sd


class ComposedFault(Object):
    """
    A fault composed of multiple simple faults. All faults have the same upper
    depth and are connected (the upper right corner equals the upper left
    corner of the next fault). This means length and area are equal to the sum
    of the individual lenghts and areas. All other parameters are equal.
    """

    def __init__(self, trace, upper_depth, lower_depth, dip, rigidity=RIGIDITY, upper_sd=None, lower_sd=None, validate=True):
        """
        faults: list of 'faults.SimpleFault' instances
        """
        if validate is True:
            pass

        self._faults = [SimpleFault(*trace[i], *trace[i+1], upper_depth,
            lower_depth, dip, rigidity, upper_sd, lower_sd, validate=False)
            for i in range(len(trace)-1)]

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


class FaultGeometryCalculator(Object):
    """
    """

    def __init__(self, n, mrd, dip, usd, lsd, validate=True):
        """
        """
        if validate is True:
            pass

        self._n = n
        self._mrd = mrd
        self._dip = dip
        self._usd = usd
        self._lsd = lsd

    def __call__(self, fault_data, validate=True):
        """
        """
        if validate is True:
            for k in fault_data:
                assert(k in ('trace', 'dip'))

        trace = fault_data['trace'].get_simplified(n=self._n)

        mrd = fault_data.get('mrd', self._mrd)
        dip = fault_data.get('dip', self._dip)
        usd = fault_data.get('usd', self._usd)
        lsd = fault_data.get('lsd', self._lsd)

        if type(mrd) is tuple:
            mrd = random.uniform(*mrd)
        if type(dip) is tuple:
            dip = random.uniform(*dip)
        if type(usd) is tuple:
            usd = random.uniform(*usd)
        if type(lsd) is tuple:
            lsd = random.uniform(*lsd)

        if self._n == 1:
            fault = SimpleFault(
                *trace[0], *trace[1], 0, mrd, dip, upper_sd=usd, lower_sd=lsd)
        else:
            fault = ComposedFault(
                trace, 0, mrd, dip, upper_sd=usd, lower_sd=lsd)

        return fault
