"""
The 'source.faults' module.
"""


import random

import matplotlib.pyplot as plt
import numpy as np

from ..apy import (Object, is_integer, is_pos_integer, is_non_neg_number,
    is_pos_number)
from .. import space2, space3
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

    def __iter__(self):
        """
        """
        for f in [self]:
            yield f

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

    def __init__(self, parts, upper_depth, lower_depth, dip, rigidity=RIGIDITY, upper_sd=None, lower_sd=None, validate=True):
        """
        parts: list of (start point, end point) tuples
        """
        if upper_sd is None:
            upper_sd = upper_depth
        if lower_sd is None:
            lower_sd = lower_depth

        if validate is True:
            assert(type(parts) is list)
            for part in parts:
                assert(len(part) == 2)
                assert(space2.are_coordinates(part[0]))
                assert(space2.are_coordinates(part[1]))
            assert(is_non_neg_number(upper_depth))
            assert(is_non_neg_number(lower_depth))
            assert(lower_depth > upper_depth)
            assert(earth.is_dip(dip))
            assert(is_pos_number(rigidity))
            assert(is_non_neg_number(upper_sd) and upper_sd >= upper_depth)
            assert(is_non_neg_number(lower_sd) and lower_sd <= lower_depth)

        self._parts = parts
        self._upper_depth = upper_depth
        self._lower_depth = lower_depth
        self._dip = dip
        self._rigidity = rigidity
        self._upper_sd = upper_sd
        self._lower_sd = lower_sd

    def __len__(self):
        """
        """
        return len(self._parts)

    def __getitem__(self, i, validate=True):
        """
        """
        if validate is True:
            assert(is_integer(i))

        f = SimpleFault(*self._parts[i][0], *self._parts[i][1],
            self._upper_depth, self._lower_depth, self._dip, self._rigidity,
            self._upper_sd, self._lower_sd, validate=False)

        return f

    def __iter__(self):
        """
        """
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, point, validate=True):
        """
        """
        p = space3.Point(*point, validate=validate)

        for s in self:
            if p in s:
                return True

        return False

    @property
    def upper_depth(self):
        """
        return: non neg number, upper depth (in m)
        """
        return self._upper_depth

    @property
    def lower_depth(self):
        """
        return: nog neg number, lower depth (in m)
        """
        return self._lower_depth

    @property
    def dip(self):
        """
        return: number, dip (angle)
        """
        return self._dip

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

    @property
    def length(self):
        """
        return: pos number
        """
        l = 0
        for p in self:
            l += p.length
        return l

    @property
    def width(self):
        """
        return: pos number
        """
        return self[0].width

    @property
    def area(self):
        """
        return: pos number
        """
        a = 0
        for p in self:
            a += p.area
        return a

    @property
    def surface(self):
        """
        return 'space2.RectangularSurface' instance
        """
        return space2.RectangularSurface(self.width, self.length)

    def get_start_and_part(self, y, validate=True):
        """
        """
        if validate is True:
            assert(is_non_neg_number(y) and y <= self.length)

        length = 0
        for p in self:
            s = length
            length += p.length
            if y <= length:
                return s, p

    def get_depths(self, shape, validate=True):
        """
        """
        d = self[0].get_discretized(shape, validate=validate)

        depths = np.repeat(
            d.cell_centers[:,0,-1][:,np.newaxis], shape[1], axis=1)

        return depths


class FaultGeometryCalculator(Object):
    """
    #TODO: implement roughness
    """

    def __init__(self, n, mrd, dip, usd, lsd, rigidity=RIGIDITY, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(n))

        self._n = n
        self._mrd = mrd
        self._dip = dip
        self._usd = usd
        self._lsd = lsd
        self._rigidity = rigidity

    def __call__(self, fault_data, validate=True):
        """
        """
        if validate is True:
            for k in fault_data:
                assert(k in ('trace', 'dip'))

        trace = fault_data['trace'].get_simplified(n=self._n)

        parts = [(trace[i], trace[i+1]) for i in range(len(trace)-1)]

        mrd = fault_data.get('mrd', self._mrd)
        dip = fault_data.get('dip', self._dip)
        usd = fault_data.get('usd', self._usd)
        lsd = fault_data.get('lsd', self._lsd)
        rigidity = fault_data.get('rigidity', self._rigidity)

        if type(mrd) is tuple:
            mrd = random.uniform(*mrd)
        if type(dip) is tuple:
            dip = random.uniform(*dip)
        if type(usd) is tuple:
            usd = random.uniform(*usd)
        if type(lsd) is tuple:
            lsd = random.uniform(*lsd)
        if type(rigidity) is tuple:
            rigidity = random.uniform(*rigidity)

        fault = ComposedFault(parts, 0, mrd, dip, rigidity, usd, lsd)

        return fault


def plot_faults(faults, colors=None, styles=None, widths=None, fill_colors=None, size=None, filespec=None, validate=True):
    """
    """
    if validate is True:
        assert(type(faults) is list)

    simple_surfaces, kwargs = [], {}

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    kwargs['colors'] = []

    if styles is not None:
        kwargs['styles'] = []
    if widths is not None:
        kwargs['widths'] = []

    if fill_colors is None:
        fill_colors = colors
    kwargs['fill_colors'] = []

    for i, fault in enumerate(faults):
        if type(fault) is SimpleFault:
            e = [fault]
        else:
            e = [f for f in fault]

        n = len(e)
        simple_surfaces.extend(e)

        if colors is not None:
            kwargs['colors'].extend([colors[i]] * n)
        if styles is not None:
            kwargs['styles'].extend([styles[i]] * n)
        if widths is not None:
            kwargs['widths'].extend([widths[i]] * n)
        if fill_colors is not None:
            kwargs['fill_colors'].extend([fill_colors[i]] * n)

    earth.plot_simple_surfaces(
        simple_surfaces, size=size, filespec=filespec, **kwargs)
