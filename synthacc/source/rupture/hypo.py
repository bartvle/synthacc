"""
The 'source.rupture.hypo' module.
"""


from abc import ABC, abstractmethod
import math

import numpy as np

from ...apy import Object, is_non_neg_number
from ... import space2
from ..faults import SimpleFault


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
            assert(type(segment) is SimpleFault)

        sin = math.sin(math.radians(segment.dip))
        xmin = float((segment.upper_sd - segment.upper_depth) / sin)
        xmax = float((segment.lower_sd - segment.upper_depth) / sin)

        return xmin, xmax


class RandomHCC(HypoCenterCalculator):
    """
    """

    def __call__(self, segment, sd, validate=True):
        """
        """
        if validate is True:
            assert(type(segment) is SimpleFault)

        xmin, xmax = self.get_xrange(segment)

        hypo = segment.surface.get_random(xmin=xmin, xmax=xmax)
        try:
            assert(is_non_neg_number(hypo[0]) and hypo[0] <= sd.w)
            assert(is_non_neg_number(hypo[1]) and hypo[1] <= sd.l)
        except Exception as e:
            print(xmin, xmax, hypo[0])
            raise e

        return hypo


class MaiEtAl2005HCC(HypoCenterCalculator):
    """
    Based on Mai et al. (2005).
    """

    def __call__(self, segment, sd, validate=True):
        """
        """
        if validate is True:
            assert(type(segment) is SimpleFault)

        indices1 = np.where(sd.values >= ((1.500*1/3) * sd.max)) ## 1/2
        indices2 = np.where(sd.values >= ((1.125*2/3) * sd.max)) ## 3/4
        xgrid1 = sd.xgrid[indices1]
        ygrid1 = sd.ygrid[indices1]
        xgrid2 = sd.xgrid[indices2]
        ygrid2 = sd.ygrid[indices2]
        distance1 = segment.length
        distance2 = segment.length
        xmin, xmax = self.get_xrange(segment)

        while (distance1 > (0.2 * segment.length) or distance2 > (0.4 * segment.length)):
            hypo = segment.surface.get_random(xmin=xmin, xmax=xmax)
            distance1 = space2.distance(hypo.x, hypo.y, xgrid1, ygrid1).min()
            distance2 = space2.distance(hypo.x, hypo.y, xgrid2, ygrid2).min()
        try:
            assert(is_non_neg_number(hypo[0]) and hypo[0] <= sd.w)
            assert(is_non_neg_number(hypo[1]) and hypo[1] <= sd.l)
        except Exception as e:
            print(xmin, xmax, hypo[0])
            raise e

        return hypo
