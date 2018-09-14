"""
The 'source.rupture.velocity' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from ...apy import Object, is_pos_number, is_2d_numeric_array
from ... import space2
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

    def __call__(self, segment, magnitude, sd, validate=True):
        """
        """
        if validate is True:
            pass
                
        velocities = np.ones(sd.shape) * (self._min_vel +
            np.random.random() * (self._max_vel - self._min_vel))

        vd = VelocityDistribution(segment.width, segment.length, velocities)

        return vd


class RandomFieldVDC(VelocityDistributionCalculator):
    """
    """

    def __init__(self, min_vel, max_vel, sd, validate=True):
        """
        """
        if validate is True:
            pass

        self._min_vel = min_vel
        self._max_vel = max_vel
        self._sd = sd

    def __call__(self, segment, magnitude, sd, validate=True):
        """
        """
        if validate is True:
            pass

        surface = segment.get_discretized(sd.shape).surface

        rfc = MaiBeroza2002RFSDC(sd.dw, sd.dl, self._sd)
        acf = rfc.get_acf()
        aw = rfc.get_aw(surface.w)
        al = rfc.get_al(surface.l)

        srfg = space2.SpatialRandomFieldGenerator(sd.w, sd.l, sd.nw, sd.nl,
            acf, aw, al, validate=validate)

        field = srfg(seed=None, validate=False)

        avg = self._min_vel + np.random.random() * (self._max_vel - self._min_vel)

        velocities = avg + field * self._sd

        vd = VelocityDistribution(surface.w, surface.l, velocities)

        return vd


# class LayeredVDC(VelocityDistributionCalculator):
#     """
#     """

#     def __init__(self, min_vel, max_vel, sd, validate=True):
#         """
#         """
#         self._min_vel = min_vel
#         self._max_vel = max_vel
#         self._sd = sd

#     def __call__(self, segment, magnitude, sd, validate=True):
#         """
#         """
#         if validate is True:
#             pass
        
#         surface = segment.get_discretized(sd.shape).surface

#         depths = np.tile(np.interp(surface.xs, [0, sd.w],
#             [segment.upper_depth, segment.lower_depth])[np.newaxis].T,
#             (1, surface.nl))

#         avg = self._min_vel + np.random.random() * (self._max_vel - self._min_vel)

#         ratios = np.interp(depths, [5000, 8000, 17000, 20000], [0.7, 1, 1, 0.7])
#         velocities = avg * ratios

#         rfc = MaiBeroza2002RFSDC(sd.dw, sd.dl, self._sd)
#         acf = rfc.get_acf()
#         aw = rfc.get_aw(surface.w)
#         al = rfc.get_al(surface.l)
#         srfg = space2.SpatialRandomFieldGenerator(sd.w, sd.l, sd.nw, sd.nl,
#             acf, aw, al, validate=validate)
#         field = srfg(seed=None, validate=False)
#         velocities += (field * self._sd)

#         vd = VelocityDistribution(sd.w, sd.l, velocities)

#         return vd


# class LayeredRandomFieldVDC(VelocityDistributionCalculator):
#     """
#     """

#     def __init__(self, min_vel, max_vel, sd, validate=True):
#         """
#         """
#         self._min_vel = min_vel
#         self._max_vel = max_vel
#         self._sd = sd

#     def __call__(self, segment, magnitude, sd, validate=True):
#         """
#         """
#         if validate is True:
#             pass

#         surface = segment.get_discretized(sd.shape).surface

#         depths = np.tile(np.interp(surface.xs, [0, surface.w],
#             [segment.upper_depth, segment.lower_depth])[np.newaxis].T,
#             (1, surface.nl))

#         avg = self._min_vel + np.random.random() * (self._max_vel - self._min_vel)

#         ratios = np.interp(depths, [5000, 8000, 17000, 20000], [0.7, 1, 1, 0.7])

#         velocities = avg * ratios

#         rfc = MaiBeroza2002RFSDC(surface.dw, surface.dl, self._sd)
#         acf = rfc.get_acf()
#         aw = rfc.get_aw(surface.w)
#         al = rfc.get_al(surface.l)

#         srfg = space2.SpatialRandomFieldGenerator(surface.w, surface.l,
#             surface.nw, surface.nl, acf, aw, al, validate=validate)

#         field = srfg(seed=None, validate=False)

#         velocities = velocities + field * self._sd

#         vd = VelocityDistribution(surface.w, surface.l, velocities)

#         return vd


# class GP2010VDC(VelocityDistributionCalculator):
#     """
#     """

#     def __init__(self, vs, validate=True):
#         """
#         """
#         self._vs = vs

#     def __call__(self, segment, magnitude, sd, validate=True):
#         """
#         """
#         if validate is True:
#             pass

#         surface = segment.get_discretized(sd.shape).surface

#         depths = np.tile(np.interp(surface.xs, [0, surface.w],
#             [segment.upper_depth, segment.lower_depth])[np.newaxis].T,
#             (1, surface.nl))

#         vs = np.interp(depths, self._vs[:,0], self._vs[:,1])

#         ratios = np.interp(depths, [5000, 8000, 17000, 20000], [0.56, 0.80, 0.80, 0.56])

#         vd = VelocityDistribution(surface.w, surface.l, vs * ratios)

#         return vd


# class GP2016VelocityDistributionGenerator(Object):
#     """
#     """

#     def __init__(self, w, l, d, upper_depth, lower_depth, vs, acf, aw, al, validate=True):
#         """
#         """
#         if validate is True:
#             pass

#         nw = int(round(w / d) // 2 * 2 + 1)
#         nl = int(round(l / d) // 2 * 2 + 1)

#         surface = space2.DiscretizedRectangularSurface(
#             w, l, nw, nl, validate=False)
            
#         depths = np.tile(np.interp(
#             surface.ws, [0, w], [upper_depth, lower_depth])[np.newaxis].T, (1, surface.nl))

#         self._surface = surface
#         self._velocities = np.interp(depths, [5000, 8000], [0.56, 0.80]) * vs

#         self._srfg = space2.SpatialRandomFieldGenerator(
#             self._surface.n, self._surface.l,
#             self._surface.nw, self._surface.nl,
#             acf, aw, al, validate=validate)

#     def __call__(self, sd, cf=1, std=0.1, validate=True):
#         """
#         """
#         if validate is True:
#             pass
            
#         field = sd.slip - sd.slip.mean()
#         field = field / np.std(field, ddof=1)

#         field = cf * field + np.sqrt(1-cf**2) * self.srfg()
#         field = field / np.std(field, ddof=1)

#         velocities = self._velocities * (1 + field * std)

#         vd = VelocityDistribution(
#             self._surface.w, self._surface.l, velocities)

#         return vd

#     @property
#     def srfg(self):
#         """
#         """
#         return self._srfg
