"""
The 'source.rupture.rake' module.
"""


from abc import ABC, abstractmethod

import numpy as np

from ...apy import Object, is_pos_number, is_2d_numeric_array
from ... import stats
from .surface import Distribution
from .slip import MaiBeroza2002RFSDC


class RakeDistribution(Distribution):
    """
    """

    LABEL = 'Rake (°)'

    def __init__(self, w, l, rake, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_number(w))
            assert(is_pos_number(l))
            assert(is_2d_numeric_array(rake))

        super().__init__(w, l, *rake.shape, validate=False)

        self._values = rake


class RakeDistributionCalculator(ABC, Object):
    """
    """

    @abstractmethod
    def __call__(self, segment, sd, rake, validate=True):
        """
        """
        pass


class ConstantRDC(RakeDistributionCalculator):
    """
    """

    def __init__(self, validate=True):
        """
        """
        if validate is True:
            pass

    def __call__(self, segment, sd, rake, validate=True):
        """
        """
        return RakeDistribution(sd.w, sd.l, np.ones_like(sd.values) * rake)


class RandomFieldRDC(RakeDistributionCalculator):
    """
    """

    def __init__(self, sd=1, validate=True):
        """
        """
        if validate is True:
            pass

        self._sd = sd

    def __call__(self, segment, sd, rake, validate=True):
        """
        """
        rfc = MaiBeroza2002RFSDC(sd.dw, sd.dl, self._sd)
        acf = rfc.get_acf()
        aw = rfc.get_aw(sd.w)
        al = rfc.get_al(sd.l)

        srfg = stats.SpatialRandomFieldGenerator(sd.w, sd.l, sd.nw, sd.nl,
            acf, aw, al, validate=validate)

        field = srfg(seed=None, validate=False)

        rake = rake + field * self._sd

        return RakeDistribution(sd.w, sd.l, rake)
