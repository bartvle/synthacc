"""
The 'source.rupture.velocity' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from ...apy import Object, is_pos_number, is_2d_numeric_array
from ... import stats
from .surface import Distribution
from .slip import MaiBeroza2002RFSDC


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

        super().__init__(w, l, *velocities.shape, validate=False)

        self._values = velocities

    @property
    def velocities(self):
        """
        """
        return self._values


class VelocityDistributionCalculator(ABC, Object):
    """
    """

    @abstractmethod
    def __call__(self, segment, magnitude, sd, validate=True):
        """
        sd: 'SlipDistribution' instance
        """
        pass


class ConstantVDC(VelocityDistributionCalculator):
    """
    """

    def __init__(self, min_vel, max_vel, validate=True):
        """
        """
        self._min_vel = min_vel
        self._max_vel = max_vel

    def __call__(self, segment, magnitude=None, sd=None, validate=True):
        """
        sd: 'SlipDistribution' instance
        """
        if validate is True:
            pass

        if sd is None:
            sd = segment
                
        velocities = np.ones(sd.shape) * (self._min_vel +
            np.random.random() * (self._max_vel - self._min_vel))

        vd = VelocityDistribution(segment.width, segment.length, velocities)

        return vd


class RandomFieldVDC(VelocityDistributionCalculator):
    """
    """

    def __init__(self, min_vel, max_vel, sd, validate=True):
        """
        sd: standard deviation
        """
        if validate is True:
            pass

        self._min_vel = min_vel
        self._max_vel = max_vel
        self._sd = sd

    def __call__(self, segment, magnitude=None, sd=None, validate=True):
        """
        sd: 'SlipDistribution' instance
        """
        if validate is True:
            pass

        if sd is None:
            sd = segment

        surface = segment.surface

        rfc = MaiBeroza2002RFSDC(sd.dw, sd.dl, self._sd)
        acf = rfc.get_acf()
        aw = rfc.get_aw(surface.w)
        al = rfc.get_al(surface.l)

        srfg = stats.SpatialRandomFieldGenerator(surface.w, surface.l,
            sd.nw, sd.nl, acf, aw, al, validate=validate)

        field = srfg(seed=None, validate=False)

        avg = self._min_vel + np.random.random() * (self._max_vel - self._min_vel)

        velocities = avg + field * self._sd

        vd = VelocityDistribution(surface.w, surface.l, velocities)

        return vd


class LayeredRandomFieldVDC(VelocityDistributionCalculator):
    """
    """

    def __init__(self, min_vel, max_vel, sd, validate=True):
        """
        sd: standard deviation
        """
        self._min_vel = min_vel
        self._max_vel = max_vel
        self._sd = sd

    def __call__(self, segment, magnitude=None, sd=None, validate=True):
        """
        sd: 'SlipDistribution' instance
        """
        if validate is True:
            pass

        if sd is None:
            sd = segment

        surface = segment.surface

        depths = np.tile(np.interp(surface.xs, [0, surface.w],
            [segment.upper_depth, segment.lower_depth])[np.newaxis].T,
            (1, surface.nl))

        avg = self._min_vel + np.random.random() * (self._max_vel - self._min_vel)

        ratios = np.interp(depths, [5000, 8000, 17000, 20000], [0.7, 1, 1, 0.7])
        velocities = avg * ratios

        rfc = MaiBeroza2002RFSDC(surface.dw, surface.dl, self._sd)
        acf = rfc.get_acf()
        aw = rfc.get_aw(surface.w)
        al = rfc.get_al(surface.l)
        srfg = stats.SpatialRandomFieldGenerator(surface.w, surface.l,
            surface.nw, surface.nl, acf, aw, al, validate=validate)
        field = srfg(seed=None, validate=False)
        velocities += (field * self._sd)

        vd = VelocityDistribution(surface.w, surface.l, velocities)

        return vd


class GP2010VDC(VelocityDistributionCalculator):
    """
    """

    def __init__(self, vs, validate=True):
        """
        """
        self._vs = vs

    def __call__(self, segment, magnitude=None, sd=None, validate=True):
        """
        """
        if validate is True:
            pass

        if sd is None:
            sd = segment

        surface = segment.surface

        depths = np.tile(np.interp(surface.xs, [0, surface.w],
            [segment.upper_depth, segment.lower_depth])[np.newaxis].T,
            (1, surface.nl))

        vs = np.interp(depths, self._vs[:,0], self._vs[:,1])

        ratios = np.interp(depths, [5000, 8000, 17000, 20000], [0.56, 0.80, 0.80, 0.56])

        vd = VelocityDistribution(surface.w, surface.l, vs * ratios)

        return vd
