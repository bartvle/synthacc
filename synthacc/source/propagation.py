"""
The 'source.propagation' module.
"""


import numpy as np
import scipy.interpolate
import skfmm

from ..apy import (Object, is_non_neg_number, is_pos_number,
    is_2d_numeric_array)
from .. import space2
from .ruptures import Surface, Distribution


class VelocityDistribution(Distribution):
    """
    """

    LABEL = 'Velocity (m/s)'

    def __init__(self, w, l, velocities, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_2d_numeric_array(velocities))
            assert(np.all(velocities > 0))

        dw = w / velocities.shape[0]
        dl = l / velocities.shape[1]

        super().__init__(w, l, dw, dl, validate=False)

        self._values = velocities

    @property
    def velocities(self):
        """
        """
        return self._values


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
            assert(type(surface) is Surface)
            assert(is_pos_number(velocity))

        self._surface = surface
        self._velocity = velocity

    def __call__(self, x, y, validate=True):
        """
        """
        if validate is True:
            assert(is_non_neg_number(x) and x <= self._surface.l)
            assert(is_non_neg_number(y) and y <= self._surface.w)

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
            # assert(vd.l % d == 0)
            # assert(vd.w % d == 0)

        nx = round(vd.l / d) + 1
        ny = round(vd.w / d) + 1
        xs = np.linspace(0, vd.l, nx)
        ys = np.linspace(0, vd.w, ny)

        i = scipy.interpolate.RectBivariateSpline(vd.ys, vd.xs, vd.velocities)
        vs = i(ys, xs)

        xgrid, ygrid = np.meshgrid(xs, ys)

        self._vd = vd
        self._d = d
        self._xs = xs
        self._ys = ys
        self._vs = vs
        self._xgrid = xgrid
        self._ygrid = ygrid

    def __call__(self, x, y, validate=True):
        """
        """
        if validate is True:
            assert(is_non_neg_number(x) and x <= self._vd.l)
            assert(is_non_neg_number(y) and y <= self._vd.w)

        distances = space2.distance(x, y, self._xgrid, self._ygrid)

        phi = np.ones(self._vs.shape)
        phi[np.unravel_index(distances.argmin(), distances.shape)] = 0

        tts = skfmm.travel_time(phi, self._vs, dx=self._d)

        i = scipy.interpolate.RectBivariateSpline(self._ys, self._xs, tts)
        tts = i(self._vd.ys, self._vd.xs)

        return TravelTimes(self._vd.w, self._vd.l, tts)
