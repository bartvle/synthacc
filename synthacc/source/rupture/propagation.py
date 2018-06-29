"""
The 'source.rupture.propagation' module.
"""


import numpy as np
import skfmm

from ...apy import (Object, is_non_neg_number, is_pos_number,
    is_2d_numeric_array)
from ... import space2
from .surface import Distribution
from .velocity import VelocityDistribution


class TravelTimes(Distribution):
    """
    """

    LABEL = 'Time (s)'

    def __init__(self, w, l, times, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_2d_numeric_array(times))
            assert(np.all(times >= 0))

        dw = w / times.shape[0]
        dl = l / times.shape[1]

        super().__init__(w, l, dw, dl, validate=False)

        self._values = times

    @property
    def times(self):
        """
        """
        return self._values


class ConstantVelocityTravelTimeCalculator(Object):
    """
    """

    def __init__(self, surface, velocity, validate=True):
        """
        """
        if validate is True:
            assert(type(surface) is space2.DiscretizedRectangularSurface)
            assert(is_pos_number(velocity))

        self._surface = surface
        self._velocity = velocity

    def __call__(self, x, y, validate=True):
        """
        The origin is in the upper left corner.
        """
        if validate is True:
            assert(is_non_neg_number(x) and x <= self._surface.w)
            assert(is_non_neg_number(y) and y <= self._surface.l)

        tts = space2.distance(x, y,
            self._surface.xgrid,
            self._surface.ygrid,
            ) / self._velocity

        return TravelTimes(self._surface.w, self._surface.l, tts)


class TravelTimeCalculator(Object):
    """
    """

    def __init__(self, vd, d, validate=True):
        """
        """
        if validate is True:
            assert(type(vd) is VelocityDistribution)
            assert(is_pos_number(d))

        ## interpolation to edge so origin can be at edge
        nw = round(vd.w / d) + 1
        nl = round(vd.l / d) + 1
        ws = np.linspace(0, vd.w, nw)
        ls = np.linspace(0, vd.l, nl)
        vs = vd.interpolate(ws, ls)

        lgrid, wgrid = np.meshgrid(ls, ws)

        self._vd = vd
        self._d = d
        self._ws = ws
        self._ls = ls
        self._vs = vs
        self._wgrid = wgrid
        self._lgrid = lgrid

    def __call__(self, x, y, ds=None, validate=True):
        """
        The origin is in the upper left corner.
        """
        if validate is True:
            assert(is_non_neg_number(x) and x <= self._vd.w)
            assert(is_non_neg_number(y) and y <= self._vd.l)

        distances = space2.distance(x, y, self._wgrid, self._lgrid)

        phi = np.ones(self._vs.shape)
        phi[np.unravel_index(distances.argmin(), distances.shape)] = 0

        tts = skfmm.travel_time(phi, self._vs, dx=self._d)
        tts = TravelTimes(self._vd.w, self._vd.l, tts)

        if ds is None:
            xs, ys = self._vd.xs, self._vd.ys
        else:
            xs, ys = ds.xs, ds.ys
        tts = tts.interpolate(xs, ys)

        tts[tts < 0] = 0

        return TravelTimes(self._vd.w, self._vd.l, tts)
