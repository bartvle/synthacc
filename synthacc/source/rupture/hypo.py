"""
The 'source.rupture.hypo' module.
"""


from abc import ABC, abstractmethod
import math

import numpy as np

from ...apy import PRECISION, Object, is_non_neg_number
from ... import space2
from ..faults import SimpleFault, ComposedFault


class HypoCenterCalculator(ABC, Object):
    """
    """

    @abstractmethod
    def __call__(self):
        """
        """
        pass

    def get_xrange(self, segment, validate=True):
        """
        """
        if validate is True:
            assert(type(segment) in [SimpleFault, ComposedFault])

        sin = math.sin(math.radians(segment.dip))
        xmin = (segment.upper_sd - segment.upper_depth) / sin
        xmax = (segment.lower_sd - segment.upper_depth) / sin

        ## It can happen that xmax is larger than segment width with a fraction
        ## smaller than 10**-PRECISION. The next two lines correct for this.
        if abs(xmax - segment.width) < 10**-PRECISION:
            xmax = segment.width

        return xmin, xmax


class RandomHCC(HypoCenterCalculator):
    """
    """

    def __call__(self, segment, sd, validate=True):
        """
        """
        if validate is True:
            assert(type(segment) in [SimpleFault, ComposedFault])

        xmin, xmax = self.get_xrange(segment)

        hypo = segment.surface.get_random(xmin=xmin, xmax=xmax)

        return hypo


class MaiEtAl2005HCC(HypoCenterCalculator):
    """
    Based on Mai et al. (2005).
    """

    def __init__(self, slip1=1/2, slip2=3/4, distance1=0.2, distance2=0.4, validate=True):
        """
        """
        if validate is True:
            pass

        self._slip1 = slip1
        self._slip2 = slip2
        self._distance1 = distance1
        self._distance2 = distance2

    def __call__(self, segment, sd, validate=True):
        """
        """
        if validate is True:
            assert(type(segment) in [SimpleFault, ComposedFault])

        indices1 = np.where(sd.values >= (self._slip1 * sd.max))
        indices2 = np.where(sd.values >= (self._slip2 * sd.max))
        xgrid1 = sd.xgrid[indices1]
        ygrid1 = sd.ygrid[indices1]
        xgrid2 = sd.xgrid[indices2]
        ygrid2 = sd.ygrid[indices2]
        distance1 = segment.length
        distance2 = segment.length
        xmin, xmax = self.get_xrange(segment)

        while (
            distance1 > (self._distance1 * segment.length) or
            distance2 > (self._distance2 * segment.length)
            ):
            hypo = segment.surface.get_random(xmin=xmin, xmax=xmax)
            distance1 = space2.distance(
                hypo.x, hypo.y, xgrid1, ygrid1).min()
            distance2 = space2.distance(
                hypo.x, hypo.y, xgrid2, ygrid2).min()

        return hypo
