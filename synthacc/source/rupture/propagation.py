"""
The 'source.rupture.propagation' module.
"""


import numpy as np
import skfmm

from ...apy import (Object, is_non_neg_number, is_pos_number,
    is_2d_numeric_array)
from ... import space2
from .models import Distribution


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


class RFVelocityDistributionGenerator(Object):
    """
    """

    def __init__(self, w, l, dw, dl, acf, aw, al, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_pos_number(dw))
            assert(is_pos_number(dl))

        nw = int(round(w / dw) // 2 * 2 + 1)
        nl = int(round(l / dl) // 2 * 2 + 1)

        dw = w / nw
        dl = l / nl

        self._surface = space2.DiscretizedRectangularSurface(
            w, l, dw, dl, validate=False)

        self._srfg = space2.SpatialRandomFieldGenerator(
            self._surface.nw, self._surface.nl,
            self._surface.dw, self._surface.dl,
            acf, aw, al, validate=validate)

    def __call__(self, velocity, std=0.1, validate=True):
        """
        """
        if validate is True:
            pass

        field = self.srfg()

        velocities = velocity * (1 + field * std)

        vd = VelocityDistribution(self._surface.w, self._surface.l, velocities)

        return vd

    @property
    def srfg(self):
        """
        """
        return self._srfg


class GP2010VelocityDistributionGenerator(Object):
    """
    """

    def __init__(self, w, l, d, upper_depth, lower_depth, vs, validate=True):
        """
        """
        if validate is True:
            pass

        surface = space2.DiscretizedRectangularSurface(w, l, d, d)

        depths = np.tile(np.interp(
            surface.ys, [0, w], [upper_depth, lower_depth])[np.newaxis].T, (1, surface.nl))

        self._surface = surface
        self._velocities = np.interp(depths, [5000, 8000], [0.56, 0.80]) * vs

    def __call__(self, validate=True):
        """
        """
        if validate is True:
            pass

        vd = VelocityDistribution(
            self._surface.w, self._surface.l, self._velocities)

        return vd


class GP2016VelocityDistributionGenerator(Object):
    """
    """

    def __init__(self, w, l, d, upper_depth, lower_depth, vs, acf, aw, al, validate=True):
        """
        """
        if validate is True:
            pass

        nw = int(round(w / d) // 2 * 2 + 1)
        nl = int(round(l / d) // 2 * 2 + 1)

        dw = w / nw
        dl = l / nl

        surface = space2.DiscretizedRectangularSurface(
            w, l, dw, dl, validate=False)

        depths = np.tile(np.interp(
            surface.ys, [0, w], [upper_depth, lower_depth])[np.newaxis].T, (1, surface.nl))

        self._surface = surface
        self._velocities = np.interp(depths, [5000, 8000], [0.56, 0.80]) * vs

        self._srfg = space2.SpatialRandomFieldGenerator(
            self._surface.nw, self._surface.nl,
            self._surface.dw, self._surface.dl,
            acf, aw, al, validate=validate)

    def __call__(self, sd, cf=1, std=0.1, validate=True):
        """
        """
        if validate is True:
            pass

        field = sd.slip - sd.slip.mean()
        field = field / np.std(field, ddof=1)

        field = cf * field + np.sqrt(1-cf**2) * self.srfg()
        field = field / np.std(field, ddof=1)
        print(field.mean(), np.std(field))

        velocities = self._velocities * (1 + field * std)

        vd = VelocityDistribution(
            self._surface.w, self._surface.l, velocities)

        return vd

    @property
    def srfg(self):
        """
        """
        return self._srfg


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

        nx = round(vd.l / d) + 1
        ny = round(vd.w / d) + 1
        xs = np.linspace(0, vd.l, nx)
        ys = np.linspace(0, vd.w, ny)

        vs = vd.interpolate(xs, ys)

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
        tts = TravelTimes(self._vd.w, self._vd.l, tts)
        tts = tts.interpolate(self._vd.xs, self._vd.ys)
        tts[tts < 0] = 0

        return TravelTimes(self._vd.w, self._vd.l, tts)
