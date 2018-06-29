"""
The 'source.rupture.velocity' module.
"""


import numpy as np

from ...apy import Object, is_pos_number, is_2d_numeric_array
from ... import space2
from .surface import Distribution


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


class GP2010VelocityDistributionCalculator(Object):
    """
    """

    def __init__(self, d, vs, validate=True):
        """
        """
        self._d = d
        self._vs = vs

    def __call__(self, surface, magnitude, sd, validate=True):
        """
        """
        vdg = GP2010VelocityDistributionGenerator(
            surface.width, surface.length, self._d, surface.upper_depth,
            surface.lower_depth, self._vs)

        return vdg

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
            surface.xs, [0, w], [upper_depth, lower_depth])[np.newaxis].T, (1, surface.nl))

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
            surface.ws, [0, w], [upper_depth, lower_depth])[np.newaxis].T, (1, surface.nl))

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

        velocities = self._velocities * (1 + field * std)

        vd = VelocityDistribution(
            self._surface.w, self._surface.l, velocities)

        return vd

    @property
    def srfg(self):
        """
        """
        return self._srfg
